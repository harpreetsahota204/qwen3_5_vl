"""
FiftyOne integration for Qwen3.5 vision-language model.

Supports both image and video understanding with full DataLoader batching.

Image operations: detect, point, classify, vqa, detect_3d
Video operations: description, temporal_localization, tracking, ocr,
                  comprehensive, custom

Architecture:
    Qwen35VLBaseConfig / Qwen35VLBaseModel
        ├── Qwen35VLImageModelConfig / Qwen35VLImageModel  (media_type="image")
        └── Qwen35VLVideoModelConfig / Qwen35VLVideoModel  (media_type="video")
"""

import logging
import json
import math
import re
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import GetItem

from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


# =============================================================================
# System prompts — image operations
# =============================================================================

DEFAULT_DETECT_SYSTEM_PROMPT = (
    "You are a helpful assistant to detect objects in images. "
    "When asked to detect elements based on a description you return bounding boxes "
    "for all elements in the form of [xmin, ymin, xmax, ymax] with the values being "
    "scaled between 0 and 1000. When there are more than one result, answer with a list "
    "of bounding boxes in the form of [[xmin, ymin, xmax, ymax], ...]. "
    "Return results as JSON array: "
    '[{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "object_name"}, ...].'
)

DEFAULT_POINT_SYSTEM_PROMPT = (
    "You are a precise object pointing assistant. When asked to point to an object in "
    "an image, you must return ONLY the exact center coordinates of that specific object "
    "as [x, y] with values scaled between 0 and 1000 (where 0,0 is the top-left corner "
    "and 1000,1000 is the bottom-right corner).\n"
    "Rules:\n"
    "1. ONLY point to objects that exactly match the description given.\n"
    "2. Do NOT point to background, empty areas, or unrelated objects.\n"
    "3. If there are multiple matching instances, return [[x1, y1], [x2, y2], ...].\n"
    "4. If no matching object is found, return an empty list [].\n"
    "5. Return ONLY the coordinate numbers, no explanations or other text.\n"
    "6. Be extremely precise — place the point at the exact visual center of each "
    "matching object."
)

DEFAULT_CLASSIFY_SYSTEM_PROMPT = (
    "You are a helpful assistant specializing in comprehensive image classification. "
    "Analyze the image and return classifications as a JSON array: "
    '[{"label": "class_name"}]. '
    "Multiple relevant classifications can be provided unless single-class output is "
    "explicitly requested."
)

DEFAULT_VQA_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide clear and concise answers to questions about "
    "images in natural language English."
)

DEFAULT_DETECT_3D_SYSTEM_PROMPT = (
    "You are a helpful assistant for 3D object detection. Detect the requested objects "
    "in the image and predict their 3D bounding boxes. "
    "Output JSON: "
    '[{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], '
    '"label": "category"}]. '
    "bbox_3d values: position in camera coordinates (meters), dimensions (meters), "
    "rotation angles as normalized fractions of pi."
)

IMAGE_OPERATIONS: Dict[str, str] = {
    "detect": DEFAULT_DETECT_SYSTEM_PROMPT,
    "point": DEFAULT_POINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFY_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "detect_3d": DEFAULT_DETECT_3D_SYSTEM_PROMPT,
}

# =============================================================================
# Video operation prompts
# =============================================================================

VIDEO_OPERATIONS: Dict[str, Dict] = {
    "comprehensive": {
        "prompt": (
            "Analyze this video comprehensively in JSON format:\n\n"
            "{\n"
            '  "summary": "Brief description of the video",\n'
            '  "objects": [{"name": "object name", "first_appears": "mm:ss.ff", '
            '"last_appears": "mm:ss.ff"}],\n'
            '  "events": [{"start": "mm:ss.ff", "end": "mm:ss.ff", '
            '"description": "event description"}],\n'
            '  "text_content": [{"start": "mm:ss.ff", "end": "mm:ss.ff", '
            '"text": "text content"}],\n'
            '  "scene_info": {"setting": "<one-word>", "time_of_day": "<one-word>", '
            '"location_type": "<one-word>"},\n'
            '  "activities": {"primary_activity": "activity name", '
            '"secondary_activities": "comma-separated activities"}\n'
            "}"
        )
    },
    "description": {
        "prompt": "Provide a detailed description of what happens in this video."
    },
    "temporal_localization": {
        "prompt": (
            "Localize activity events in the video. Output start and end timestamp "
            "for each event.\nProvide in JSON format with 'mm:ss.ff' format:\n"
            '[{"start": "mm:ss.ff", "end": "mm:ss.ff", "description": "..."}]'
        )
    },
    "tracking": {
        "prompt": (
            "Track all objects in this video. For each frame where objects appear, "
            "provide:\n"
            "- time: timestamp (mm:ss.ff)\n"
            "- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale\n"
            "- label: object label\n"
            'Output in JSON: [{"time": "mm:ss.ff", "bbox_2d": [...], "label": "..."}, ...]'
        )
    },
    "ocr": {
        "prompt": (
            "Extract all text appearing in this video. For each text instance, provide:\n"
            "- time: timestamp (mm:ss.ff)\n"
            "- text: the actual text content\n"
            "- bbox_2d: bounding box as [x_min, y_min, x_max, y_max] in 0-1000 scale\n"
            'Output in JSON: [{"time": "mm:ss.ff", "text": "...", "bbox_2d": [...]}, ...]'
        )
    },
    "custom": {
        "prompt": None  # user-supplied via custom_prompt config param
    },
}


# =============================================================================
# Device helper
# =============================================================================

def get_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =============================================================================
# Shared GetItem
# =============================================================================

class Qwen35VLGetItem(GetItem):
    """Lightweight DataLoader transform shared by image and video models.

    Extracts filepath, optional per-sample prompt, and sample metadata.
    Heavy processing (vision encoding, model forward) happens in predict_all.
    """

    @property
    def required_keys(self) -> List[str]:
        # Only include fields that are guaranteed to exist on every sample.
        # "prompt_field" is a logical key that enters field_mapping only when
        # the user explicitly sets needs_fields = {"prompt_field": "my_field"}.
        # Including it here would cause FiftyOne to try select_fields(["prompt_field"])
        # which fails when no such field exists on the dataset.
        return ["filepath", "metadata"]

    def __call__(self, sample_dict: dict) -> dict:
        return {
            "filepath": sample_dict["filepath"],
            "prompt": sample_dict.get("prompt_field"),
            "metadata": sample_dict.get("metadata"),
        }


# =============================================================================
# Base config
# =============================================================================

class Qwen35VLBaseConfig(fout.TorchImageModelConfig):
    """Shared configuration: model path and text generation parameters."""

    def __init__(self, d: dict):
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        super().__init__(d)

        self.model_path = self.parse_string(d, "model_path", default="Qwen/Qwen3.5-9B")
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=8192)
        self.do_sample = self.parse_bool(d, "do_sample", default=True)
        self.temperature = self.parse_number(d, "temperature", default=1.0)
        self.top_p = self.parse_number(d, "top_p", default=0.95)
        self.top_k = self.parse_number(d, "top_k", default=20)
        self.min_p = self.parse_number(d, "min_p", default=0.0)
        self.repetition_penalty = self.parse_number(d, "repetition_penalty", default=1.0)


# =============================================================================
# Base model
# =============================================================================

class Qwen35VLBaseModel(fom.Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin):
    """Shared base for image and video Qwen3.5-VL zoo models.

    Provides:
    - Model loading (_load_model) with hardware-appropriate dtype and attention
    - Reasoning extraction (_extract_reasoning)
    - JSON extraction (_extract_json)
    - All FiftyOne batching boilerplate
    """

    def __init__(self, config: Qwen35VLBaseConfig):
        fom.SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)

        self._preprocess = False
        self.config = config
        self.device = get_device()
        self._fields: dict = {}

        # Lazy loading
        self._model: Optional[Qwen3_5ForConditionalGeneration] = None
        self._processor: Optional[AutoProcessor] = None

        logger.info(f"Initialized {self.__class__.__name__} (device: {self.device})")

    # -------------------------------------------------------------------------
    # Required Model / SamplesMixin properties
    # -------------------------------------------------------------------------

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: bool):
        self._preprocess = value

    @property
    def ragged_batches(self) -> bool:
        # Must be False to enable DataLoader batching even for variable-size inputs.
        # Variable sizes are handled by the identity collate_fn.
        return False

    @property
    def needs_fields(self) -> dict:
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields: dict):
        self._fields = fields

    # -------------------------------------------------------------------------
    # TorchModelMixin — custom collation (prevents np.stack on paths/videos)
    # -------------------------------------------------------------------------

    @property
    def has_collate_fn(self) -> bool:
        return True

    @property
    def collate_fn(self):
        def identity_collate(batch):
            return batch
        return identity_collate

    # -------------------------------------------------------------------------
    # SupportsGetItem
    # -------------------------------------------------------------------------

    def build_get_item(self, field_mapping=None) -> Qwen35VLGetItem:
        return Qwen35VLGetItem(field_mapping=field_mapping)

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load Qwen3.5 model and processor.

        dtype:
            bfloat16 on Ampere+ GPUs (compute capability >= 8.0), else "auto".
        """
        logger.info(f"Loading Qwen3.5 model from {self.config.model_path}")

        model_kwargs: dict = {"device_map": self.device}

        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] >= 8:
                model_kwargs["torch_dtype"] = torch.bfloat16
                logger.info("Using bfloat16 (Ampere+ GPU detected)")
            else:
                model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = "auto"

        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            **model_kwargs,
        ).eval()

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully")

    # -------------------------------------------------------------------------
    # Shared utilities
    # -------------------------------------------------------------------------

    def _extract_reasoning(self, text: str) -> Tuple[Optional[str], str]:
        """Split model output on </think>.

        Returns:
            (reasoning_or_None, prediction_text)
        """
        if "</think>" in text:
            parts = text.split("</think>", 1)
            reasoning = parts[0].strip()
            prediction = parts[1].strip()
            return (reasoning if reasoning else None), prediction
        return None, text

    def _extract_json(self, text: str) -> Optional[Any]:
        """Extract and parse JSON from model output.

        Handles markdown-fenced ```json blocks and raw JSON.
        Returns the parsed object or None on failure.
        """
        # Try markdown fence first
        json_match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_bracket = text.find("[")
            start_brace = text.find("{")

            if start_bracket == -1 and start_brace == -1:
                return None

            if start_bracket != -1 and (
                start_brace == -1 or start_bracket < start_brace
            ):
                end = text.rfind("]")
                json_str = text[start_bracket : end + 1] if end != -1 else None
            else:
                end = text.rfind("}")
                json_str = text[start_brace : end + 1] if end != -1 else None

        if not json_str:
            return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return None

    # -------------------------------------------------------------------------
    # Generation parameter properties (shared convenience accessors)
    # -------------------------------------------------------------------------

    @property
    def max_new_tokens(self) -> int:
        return self.config.max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int):
        self.config.max_new_tokens = value

    @property
    def do_sample(self) -> bool:
        return self.config.do_sample

    @do_sample.setter
    def do_sample(self, value: bool):
        self.config.do_sample = value

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.config.temperature = value

    @property
    def top_p(self) -> float:
        return self.config.top_p

    @top_p.setter
    def top_p(self, value: float):
        self.config.top_p = value

    @property
    def top_k(self) -> int:
        return self.config.top_k

    @top_k.setter
    def top_k(self, value: int):
        self.config.top_k = value

    @property
    def min_p(self) -> float:
        return self.config.min_p

    @min_p.setter
    def min_p(self, value: float):
        self.config.min_p = value

    @property
    def repetition_penalty(self) -> float:
        return self.config.repetition_penalty

    @repetition_penalty.setter
    def repetition_penalty(self, value: float):
        self.config.repetition_penalty = value

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False


# =============================================================================
# Image config
# =============================================================================

class Qwen35VLImageModelConfig(Qwen35VLBaseConfig):
    """Configuration for the Qwen3.5-VL image model."""

    def __init__(self, d: dict):
        super().__init__(d)

        self.operation = self.parse_string(d, "operation", default="vqa")
        if self.operation not in IMAGE_OPERATIONS:
            raise ValueError(
                f"Invalid image operation: '{self.operation}'. "
                f"Must be one of {list(IMAGE_OPERATIONS.keys())}"
            )

        self.prompt = self.parse_string(d, "prompt", default=None)
        self.system_prompt = self.parse_string(d, "system_prompt", default=None)

        # detect_3d-specific camera params
        self.camera_intrinsics_field = self.parse_string(
            d, "camera_intrinsics_field", default=None
        )
        self.fov = self.parse_number(d, "fov", default=60.0)


# =============================================================================
# Image model
# =============================================================================

class Qwen35VLImageModel(Qwen35VLBaseModel):
    """FiftyOne zoo model for Qwen3.5-VL image understanding.

    Operations:
        detect       — fo.Detections (bbox_2d, 0-1000 scale)
        point        — fo.Keypoints  ([x,y] or [[x,y],...], 0-1000 scale)
        classify     — fo.Classifications
        vqa          — str
        detect_3d    — fo.Detections with location/dimensions/rotation + bounding_box

    All coordinates are in 0-1000 scale from the model, normalized to [0,1] on output.
    Reasoning chains (text before </think>) are stored as a 'reasoning' attribute
    on each label.
    """

    @property
    def media_type(self) -> str:
        return "image"

    # -------------------------------------------------------------------------
    # Operation / prompt properties
    # -------------------------------------------------------------------------

    @property
    def operation(self) -> str:
        return self.config.operation

    @operation.setter
    def operation(self, value: str):
        if value not in IMAGE_OPERATIONS:
            raise ValueError(
                f"Invalid image operation: '{value}'. "
                f"Must be one of {list(IMAGE_OPERATIONS.keys())}"
            )
        self.config.operation = value

    @property
    def system_prompt(self) -> str:
        if self.config.system_prompt is not None:
            return self.config.system_prompt
        return IMAGE_OPERATIONS[self.config.operation]

    @system_prompt.setter
    def system_prompt(self, value: Optional[str]):
        self.config.system_prompt = value

    @property
    def prompt(self) -> Optional[str]:
        return self.config.prompt

    @prompt.setter
    def prompt(self, value: Optional[str]):
        self.config.prompt = value

    @property
    def camera_intrinsics_field(self) -> Optional[str]:
        return self.config.camera_intrinsics_field

    @camera_intrinsics_field.setter
    def camera_intrinsics_field(self, value: Optional[str]):
        self.config.camera_intrinsics_field = value

    @property
    def fov(self) -> float:
        return self.config.fov

    @fov.setter
    def fov(self, value: float):
        self.config.fov = value

    # -------------------------------------------------------------------------
    # Message building
    # -------------------------------------------------------------------------

    def _build_image_message(self, filepath: str, prompt: str) -> list:
        """Build the messages list for image inference."""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": filepath},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _run_image_inference(self, messages: list) -> str:
        """Run image inference using native single-call apply_chat_template.

        Uses tokenize=True, return_dict=True so the processor handles both
        tokenization and vision encoding in one call.

        Returns:
            Generated output text (str)
        """
        if self._model is None:
            self._load_model()

        device = next(self._model.parameters()).device

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        gen_kwargs: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty
        }
        if self.config.do_sample:
            gen_kwargs.update(
                {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                }
            )

        with torch.no_grad():
            raw_ids = self._model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs.input_ids, raw_ids)
        ]

        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # -------------------------------------------------------------------------
    # Output parsing — dispatcher
    # -------------------------------------------------------------------------

    def _parse_image_output(
        self,
        text: str,
        filepath: str,
        sample,
    ):
        """Extract reasoning then dispatch to the appropriate converter."""
        reasoning, prediction = self._extract_reasoning(text)

        if self.config.operation == "vqa":
            return prediction.strip()

        if self.config.operation == "detect":
            return self._to_detections(self._extract_json(prediction), reasoning)

        if self.config.operation == "point":
            return self._to_keypoints(self._extract_json(prediction), reasoning)

        if self.config.operation == "classify":
            return self._to_classifications(self._extract_json(prediction), reasoning)

        if self.config.operation == "detect_3d":
            cam_params = self._get_camera_params(sample, filepath)
            return self._to_3d_detections(
                self._extract_json(prediction), cam_params, reasoning
            )

        logger.warning(f"Unknown operation: {self.config.operation}")
        return prediction.strip()

    # -------------------------------------------------------------------------
    # Output converters — 2D
    # -------------------------------------------------------------------------

    def _to_detections(
        self, boxes, reasoning: Optional[str] = None
    ) -> fo.Detections:
        """Convert model bbox_2d output to fo.Detections.

        Accepts:
            - List of dicts: [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}]
            - Raw list of lists: [[x1,y1,x2,y2], ...]
            - Single dict or list

        Coordinates are in 0-1000 scale → normalized to [0,1].
        """
        if not boxes:
            return fo.Detections(detections=[])

        if isinstance(boxes, dict):
            boxes = [boxes]
        elif not isinstance(boxes, list):
            return fo.Detections(detections=[])

        detections = []
        for box in boxes:
            try:
                if isinstance(box, dict):
                    bbox = box.get("bbox_2d", box.get("bbox"))
                    label = str(box.get("label", "object"))
                elif isinstance(box, list) and len(box) == 4:
                    bbox = box
                    label = "object"
                else:
                    continue

                if not bbox or len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                det = fo.Detection(
                    label=label,
                    bounding_box=[x1 / 1000, y1 / 1000, (x2 - x1) / 1000, (y2 - y1) / 1000],
                )
                if reasoning is not None:
                    det["reasoning"] = reasoning
                detections.append(det)

            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")

        return fo.Detections(detections=detections)

    def _to_keypoints(
        self, points, reasoning: Optional[str] = None
    ) -> fo.Keypoints:
        """Convert model point output to fo.Keypoints.

        Handles:
            - Single point: [x, y]
            - Multiple points: [[x1,y1], [x2,y2], ...]
            - Dict format: [{"point_2d": [x,y], "label": "..."}]  (fallback)

        Coordinates in 0-1000 scale → normalized to [0,1].
        """
        if not points and points != 0:
            return fo.Keypoints(keypoints=[])

        if not isinstance(points, list):
            return fo.Keypoints(keypoints=[])

        # Detect single [x, y] pair
        if len(points) == 2 and all(isinstance(v, (int, float)) for v in points):
            points = [points]

        keypoints = []
        for pt in points:
            try:
                if isinstance(pt, list) and len(pt) == 2:
                    x, y = float(pt[0]), float(pt[1])
                    label = "point"
                elif isinstance(pt, dict):
                    coords = pt.get("point_2d", pt.get("point"))
                    if not coords or len(coords) < 2:
                        continue
                    x, y = float(coords[0]), float(coords[1])
                    label = str(pt.get("label", "point"))
                else:
                    continue

                kp = fo.Keypoint(label=label, points=[[x / 1000, y / 1000]])
                if reasoning is not None:
                    kp["reasoning"] = reasoning
                keypoints.append(kp)

            except Exception as e:
                logger.debug(f"Error processing point {pt}: {e}")

        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(
        self, classes, reasoning: Optional[str] = None
    ) -> fo.Classifications:
        """Convert model classification output to fo.Classifications."""
        if not classes:
            return fo.Classifications(classifications=[])

        if isinstance(classes, dict):
            classes = [classes]
        elif not isinstance(classes, list):
            return fo.Classifications(classifications=[])

        classifications = []
        for cls in classes:
            try:
                if isinstance(cls, dict):
                    label = str(cls.get("label", ""))
                elif isinstance(cls, str):
                    label = cls
                else:
                    continue

                if not label:
                    continue

                c = fo.Classification(label=label)
                if reasoning is not None:
                    c["reasoning"] = reasoning
                classifications.append(c)

            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")

        return fo.Classifications(classifications=classifications)

    # -------------------------------------------------------------------------
    # 3D detection helpers
    # -------------------------------------------------------------------------

    def _get_camera_params(self, sample, filepath: str) -> dict:
        """Resolve camera intrinsics for 3D projection.

        Priority:
            1. Sample field named by camera_intrinsics_field (must have fx,fy,cx,cy)
            2. Generated pseudo params from image size + fov
        """
        if self.config.camera_intrinsics_field and sample is not None:
            try:
                cam = sample.get_field(self.config.camera_intrinsics_field)
                if cam is not None and all(k in cam for k in ("fx", "fy", "cx", "cy")):
                    return cam
            except Exception:
                pass

        return self._generate_camera_params(filepath)

    def _generate_camera_params(self, filepath: str) -> dict:
        """Generate pseudo camera params from image size and configured fov.

        Matches 3d_grounding.py exactly: fx derived from width, fy from height.
        """
        from PIL import Image as PILImage

        img = PILImage.open(filepath)
        w, h = img.size
        half_fov = math.radians(self.config.fov) / 2
        fx = round(w / (2 * math.tan(half_fov)), 2)
        fy = round(h / (2 * math.tan(half_fov)), 2)
        return {"fx": fx, "fy": fy, "cx": round(w / 2, 2), "cy": round(h / 2, 2)}

    def _project_3d_corners(self, bbox_3d: list, cam_params: dict) -> list:
        """Project 3D bounding box corners to 2D image coordinates.

        Follows 3d_grounding.py convention exactly:
        - bbox_3d[6] → pitch, [7] → yaw, [8] → roll  (despite format docs saying roll,pitch,yaw)
        - Angle values are fractions of π: radians = model_value * π
        - Rotation applied: pitch → yaw → roll

        Args:
            bbox_3d: [cx, cy, cz, sx, sy, sz, a0, a1, a2]
            cam_params: {"fx", "fy", "cx", "cy"}

        Returns:
            List of [x_2d, y_2d] for corners with Z > 0
        """
        cx, cy, cz = bbox_3d[0], bbox_3d[1], bbox_3d[2]
        sx, sy, sz = bbox_3d[3], bbox_3d[4], bbox_3d[5]
        # Unpack as pitch, yaw, roll per 3d_grounding.py line 93
        pitch = bbox_3d[6] * math.pi
        yaw = bbox_3d[7] * math.pi
        roll = bbox_3d[8] * math.pi

        hx, hy, hz = sx / 2, sy / 2, sz / 2
        local_corners = [
            [ hx,  hy,  hz],
            [ hx,  hy, -hz],
            [ hx, -hy,  hz],
            [ hx, -hy, -hz],
            [-hx,  hy,  hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [-hx, -hy, -hz],
        ]

        def rotate_xyz(pt, _pitch, _yaw, _roll):
            x0, y0, z0 = pt
            # Pitch (around x-axis)
            x1 = x0
            y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
            z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)
            # Yaw (around y-axis)
            x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
            y2 = y1
            z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)
            # Roll (around z-axis)
            x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
            y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
            z3 = z2
            return [x3, y3, z3]

        img_corners = []
        for corner in local_corners:
            rotated = rotate_xyz(corner, pitch, yaw, roll)
            X = rotated[0] + cx
            Y = rotated[1] + cy
            Z = rotated[2] + cz
            if Z > 0:
                x_2d = cam_params["fx"] * (X / Z) + cam_params["cx"]
                y_2d = cam_params["fy"] * (Y / Z) + cam_params["cy"]
                img_corners.append([x_2d, y_2d])

        return img_corners

    def _to_3d_detections(
        self, items, cam_params: dict, reasoning: Optional[str] = None
    ) -> fo.Detections:
        """Convert bbox_3d output to fo.Detections with 3D and 2D attributes.

        Each detection stores:
            location, dimensions, rotation  — for FiftyOne 3D visualizer
            bounding_box                    — 2D AABB of projected corners (regular viewer)

        rotation is stored in radians (model value * pi).
        bounding_box is omitted if fewer than 4 corners project successfully.
        """
        if not items:
            return fo.Detections(detections=[])

        if isinstance(items, dict):
            items = [items]
        elif not isinstance(items, list):
            return fo.Detections(detections=[])

        # Image dims from principal point (cx = w/2, cy = h/2)
        img_w = cam_params["cx"] * 2
        img_h = cam_params["cy"] * 2

        detections = []
        for item in items:
            try:
                if isinstance(item, dict):
                    bbox_3d = item.get("bbox_3d")
                    label = str(item.get("label", "object"))
                elif isinstance(item, list) and len(item) >= 9:
                    bbox_3d = item
                    label = "object"
                else:
                    continue

                if not bbox_3d or len(bbox_3d) < 9:
                    continue

                bbox_3d = [float(v) for v in bbox_3d[:9]]
                cx_3d, cy_3d, cz_3d = bbox_3d[0], bbox_3d[1], bbox_3d[2]
                sx, sy, sz = bbox_3d[3], bbox_3d[4], bbox_3d[5]
                # Store rotation in radians (model value * pi)
                rotation_rad = [bbox_3d[6] * math.pi, bbox_3d[7] * math.pi, bbox_3d[8] * math.pi]

                det = fo.Detection(
                    label=label,
                    location=[cx_3d, cy_3d, cz_3d],
                    dimensions=[sx, sy, sz],
                    rotation=rotation_rad,
                )

                # Project corners to get 2D bounding box
                img_corners = self._project_3d_corners(bbox_3d, cam_params)
                if len(img_corners) >= 4:
                    xs = [c[0] for c in img_corners]
                    ys = [c[1] for c in img_corners]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    det.bounding_box = [
                        x_min / img_w,
                        y_min / img_h,
                        (x_max - x_min) / img_w,
                        (y_max - y_min) / img_h,
                    ]

                if reasoning is not None:
                    det["reasoning"] = reasoning

                detections.append(det)

            except Exception as e:
                logger.debug(f"Error processing 3D detection {item}: {e}")

        return fo.Detections(detections=detections)

    # -------------------------------------------------------------------------
    # predict / predict_all
    # -------------------------------------------------------------------------

    def predict(self, arg, sample=None):
        """Single-sample inference.

        Accepts a filepath string, image reader object, or dict from GetItem.
        """
        if isinstance(arg, dict):
            batch_item = arg
        else:
            if isinstance(arg, str):
                filepath = arg
            elif hasattr(arg, "inpath"):
                filepath = arg.inpath
            elif hasattr(arg, "path"):
                filepath = arg.path
            else:
                filepath = str(arg)

            prompt = None
            if sample is not None and "prompt_field" in self._fields:
                field_name = self._fields["prompt_field"]
                if sample.has_field(field_name):
                    prompt = sample.get_field(field_name)

            batch_item = {"filepath": filepath, "prompt": prompt, "metadata": None}

        results = self.predict_all([batch_item], samples=[sample] if sample else None)
        return results[0]

    def predict_all(self, batch: list, samples=None) -> list:
        """Batch image inference.

        Args:
            batch:   List of dicts with "filepath" and "prompt" keys (from GetItem)
            samples: Optional list of FiftyOne samples (for camera intrinsics lookup)

        Returns:
            List of FiftyOne labels, one per sample
        """
        if not batch:
            return []

        if self._model is None:
            self._load_model()

        results = []
        for i, item in enumerate(batch):
            filepath = item["filepath"]
            sample = samples[i] if samples else None

            prompt = item.get("prompt") or self.config.prompt
            if not prompt:
                raise ValueError(
                    f"No prompt provided for image operation '{self.config.operation}'. "
                    "Set model.prompt or pass prompt_field to apply_model()."
                )

            messages = self._build_image_message(filepath, prompt)
            output_text = self._run_image_inference(messages)
            label = self._parse_image_output(output_text, filepath, sample)
            results.append(label)

        return results


# =============================================================================
# Video config
# =============================================================================

class Qwen35VLVideoModelConfig(Qwen35VLBaseConfig):
    """Configuration for the Qwen3.5-VL video model."""

    def __init__(self, d: dict):
        super().__init__(d)

        # Video processing
        self.total_pixels = self.parse_number(d, "total_pixels", default=2048 * 32 * 32)
        self.min_pixels = self.parse_number(d, "min_pixels", default=64 * 32 * 32)
        self.max_frames = self.parse_number(d, "max_frames", default=120)
        self.sample_fps = self.parse_number(d, "sample_fps", default=10)
        self.image_patch_size = self.parse_number(d, "image_patch_size", default=16)

        # Operation
        self.operation = self.parse_string(d, "operation", default="comprehensive")
        if self.operation not in VIDEO_OPERATIONS:
            raise ValueError(
                f"Invalid video operation: '{self.operation}'. "
                f"Must be one of {list(VIDEO_OPERATIONS.keys())}"
            )

        self.custom_prompt = self.parse_string(d, "custom_prompt", default=None)

        if self.operation == "custom" and self.custom_prompt is None:
            raise ValueError("custom_prompt is required when operation='custom'")
        if self.operation != "custom" and self.custom_prompt is not None:
            raise ValueError("custom_prompt is only allowed when operation='custom'")


# =============================================================================
# Video model
# =============================================================================

class Qwen35VLVideoModel(Qwen35VLBaseModel):
    """FiftyOne zoo model for Qwen3.5-VL video understanding.

    Operations:
        description          — str (in "summary" field)
        temporal_localization — fo.TemporalDetections (in "events" field)
        tracking             — frame-level fo.Detections (in "objects" field)
        ocr                  — frame-level fo.Detections (in "text_content" field)
        comprehensive        — mixed sample-level and frame-level labels
        custom               — str (in "result" field), prompt per-sample or global

    Requires dataset.compute_metadata() for temporal_localization, tracking, ocr,
    and comprehensive operations.
    """

    @property
    def media_type(self) -> str:
        return "video"

    # -------------------------------------------------------------------------
    # Operation / prompt properties
    # -------------------------------------------------------------------------

    @property
    def operation(self) -> str:
        return self.config.operation

    @operation.setter
    def operation(self, value: str):
        if value not in VIDEO_OPERATIONS:
            raise ValueError(
                f"Invalid video operation: '{value}'. "
                f"Must be one of {list(VIDEO_OPERATIONS.keys())}"
            )
        self.config.operation = value

    @property
    def prompt(self) -> Optional[str]:
        if self.config.operation == "custom":
            return self.config.custom_prompt
        return VIDEO_OPERATIONS[self.config.operation]["prompt"]

    @prompt.setter
    def prompt(self, value: str):
        if self.config.operation != "custom":
            raise ValueError(
                "Cannot set prompt directly for predefined operations. "
                "Use operation='custom' and set custom_prompt."
            )
        self.config.custom_prompt = value

    @property
    def custom_prompt(self) -> Optional[str]:
        if self.config.operation != "custom":
            raise ValueError("custom_prompt is only accessible when operation='custom'")
        return self.config.custom_prompt

    @custom_prompt.setter
    def custom_prompt(self, value: str):
        if self.config.operation != "custom":
            raise ValueError("custom_prompt is only allowed when operation='custom'")
        self.config.custom_prompt = value

    # Video processing properties

    @property
    def total_pixels(self) -> int:
        return self.config.total_pixels

    @total_pixels.setter
    def total_pixels(self, value: int):
        self.config.total_pixels = value

    @property
    def min_pixels(self) -> int:
        return self.config.min_pixels

    @min_pixels.setter
    def min_pixels(self, value: int):
        self.config.min_pixels = value

    @property
    def max_frames(self) -> int:
        return self.config.max_frames

    @max_frames.setter
    def max_frames(self, value: int):
        self.config.max_frames = value

    @property
    def sample_fps(self) -> int:
        return self.config.sample_fps

    @sample_fps.setter
    def sample_fps(self, value: int):
        self.config.sample_fps = value

    @property
    def image_patch_size(self) -> int:
        return self.config.image_patch_size

    @image_patch_size.setter
    def image_patch_size(self, value: int):
        self.config.image_patch_size = value

    # -------------------------------------------------------------------------
    # Message building
    # -------------------------------------------------------------------------

    def _build_video_message(self, filepath: str, prompt: str) -> list:
        """Build the messages list for video inference."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "video": filepath,
                        "total_pixels": self.config.total_pixels,
                        "min_pixels": self.config.min_pixels,
                        "max_frames": self.config.max_frames,
                        "sample_fps": self.config.sample_fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def _run_video_inference(self, messages: list) -> Tuple[str, dict]:
        """Run video inference using the two-step process_vision_info approach.

        Returns:
            (output_text, video_metadata)
        """
        if self._model is None:
            self._load_model()

        device = next(self._model.parameters()).device

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=self.config.image_patch_size,
            return_video_metadata=True,
        )

        if video_inputs:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs = list(video_inputs)
            video_metadatas = list(video_metadatas)
        else:
            video_metadatas = [{}]

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt",
        ).to(device)

        gen_kwargs: dict = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if self.config.do_sample:
            gen_kwargs.update(
                {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "min_p": self.config.min_p,
                }
            )

        with torch.no_grad():
            raw_ids = self._model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out[len(inp) :]
            for inp, out in zip(inputs["input_ids"], raw_ids)
        ]

        output_text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return output_text, video_metadatas[0]

    # -------------------------------------------------------------------------
    # Output parsing — dispatcher
    # -------------------------------------------------------------------------

    def _parse_video_output(
        self, output_text: str, sample, video_metadata: Optional[dict] = None
    ) -> dict:
        """Parse model output into FiftyOne labels based on current operation."""
        if self.config.operation == "description":
            return {"summary": output_text}

        if self.config.operation == "custom":
            return {"result": output_text}

        json_data = self._extract_json(output_text)

        if self.config.operation == "temporal_localization":
            return self._parse_temporal_only(json_data, sample)

        if self.config.operation == "tracking":
            return self._parse_tracking_only(json_data, sample, video_metadata)

        if self.config.operation == "ocr":
            return self._parse_ocr_only(json_data, sample, video_metadata)

        if self.config.operation == "comprehensive":
            return self._parse_comprehensive(json_data, sample, video_metadata)

        logger.warning(f"Unknown video operation: {self.config.operation}")
        return {"summary": output_text}

    # -------------------------------------------------------------------------
    # Operation-specific parsers
    # -------------------------------------------------------------------------

    def _parse_temporal_only(self, json_data, sample) -> dict:
        if isinstance(json_data, list):
            items = json_data or []
        elif isinstance(json_data, dict) and "events" in json_data:
            items = json_data["events"] or []
        else:
            if json_data:
                logger.warning("Expected list or dict with 'events' for temporal_localization")
            items = []

        if not items:
            return {"events": fol.TemporalDetections(detections=[])}

        detections = self._parse_temporal_detections(items, sample, "events")
        return {"events": detections or fol.TemporalDetections(detections=[])}

    def _parse_tracking_only(
        self, json_data, sample, video_metadata: Optional[dict] = None
    ) -> dict:
        if isinstance(json_data, list):
            items = json_data or []
        elif isinstance(json_data, dict) and "objects" in json_data:
            items = json_data["objects"] or []
        else:
            items = []

        if not items:
            return {"objects": fol.Detections(detections=[])}

        frame_detections = self._parse_frame_detections(
            items, sample, text_key=None, video_metadata=video_metadata
        )
        if not frame_detections:
            return {"objects": fol.Detections(detections=[])}

        return {frame_num: {"objects": dets} for frame_num, dets in frame_detections.items()}

    def _parse_ocr_only(
        self, json_data, sample, video_metadata: Optional[dict] = None
    ) -> dict:
        if isinstance(json_data, list):
            items = json_data or []
        elif isinstance(json_data, dict) and "text_content" in json_data:
            items = json_data["text_content"] or []
        else:
            items = []

        if not items:
            return {"text_content": fol.Detections(detections=[])}

        frame_detections = self._parse_frame_detections(
            items, sample, text_key="text", video_metadata=video_metadata
        )
        if not frame_detections:
            return {"text_content": fol.Detections(detections=[])}

        return {
            frame_num: {"text_content": dets}
            for frame_num, dets in frame_detections.items()
        }

    def _parse_comprehensive(
        self, json_data, sample, video_metadata: Optional[dict] = None
    ) -> dict:
        if not json_data:
            return {"summary": "No structured output from model"}

        if isinstance(json_data, list):
            logger.warning("Expected dict for comprehensive, got list")
            return {"summary": "Invalid output format: received list instead of dict"}

        labels: dict = {}
        for key, value in json_data.items():
            if isinstance(value, str):
                labels[key] = value
            elif isinstance(value, dict) and self._is_simple_dict(value):
                self._parse_dict_value(key, value, labels)
            elif isinstance(value, list) and value:
                self._parse_list_value(key, value, labels, sample, video_metadata)

        return labels

    # -------------------------------------------------------------------------
    # Label builders
    # -------------------------------------------------------------------------

    def _is_simple_dict(self, value: dict) -> bool:
        return all(isinstance(v, (str, int, float, bool)) for v in value.values())

    def _parse_dict_value(self, key: str, value: dict, labels: dict):
        for subkey, subvalue in value.items():
            field_name = f"{key}_{subkey}"
            if subkey.endswith("activities"):
                if isinstance(subvalue, str):
                    items = [s.strip().capitalize() for s in subvalue.split(",") if s.strip()]
                    labels[field_name] = fol.Classifications(
                        classifications=[fol.Classification(label=i) for i in items]
                    )
                else:
                    labels[field_name] = fol.Classifications(
                        classifications=[fol.Classification(label=str(subvalue).capitalize())]
                    )
            else:
                labels[field_name] = fol.Classification(label=str(subvalue).capitalize())

    def _parse_list_value(
        self, key: str, value: list, labels: dict, sample, video_metadata=None
    ):
        first = value[0]
        patterns = [
            (["start", "end", "description"], self._parse_temporal_detections, "events"),
            (["name", "first_appears", "last_appears"], self._parse_temporal_detections, "objects"),
            (["start", "end", "text"], self._parse_temporal_detections, "text"),
            (["time", "bbox_2d", "label"], self._parse_frame_detections, None),
            (["time", "text", "bbox_2d"], self._parse_frame_detections, "text"),
        ]

        for required_keys, parser, label_type in patterns:
            if all(k in first for k in required_keys):
                if parser == self._parse_frame_detections:
                    frame_labels = parser(value, sample, label_type, video_metadata=video_metadata)
                    self._merge_frame_labels(labels, frame_labels, key)
                else:
                    detections = parser(value, sample, label_type)
                    if detections:
                        labels[key] = detections
                return

    def _parse_temporal_detections(
        self, items: list, sample, label_type: str
    ) -> Optional[fol.TemporalDetections]:
        detections = []
        for item in items:
            if label_type == "events":
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("description", "event")).capitalize()
            elif label_type == "objects":
                start = item.get("first_appears", "00:00.00")
                end = item.get("last_appears", "00:00.00")
                label = str(item.get("name", "object")).capitalize()
            else:  # text
                start = item.get("start", "00:00.00")
                end = item.get("end", "00:00.00")
                label = str(item.get("text", "text")).capitalize()

            start_sec = self._timestamp_to_seconds(start)
            end_sec = self._timestamp_to_seconds(end)

            detection = fol.TemporalDetection.from_timestamps(
                [start_sec, end_sec], label=label, sample=sample
            )
            detections.append(detection)

        return fol.TemporalDetections(detections=detections) if detections else None

    def _parse_frame_detections(
        self,
        items: list,
        sample,
        text_key: Optional[str] = None,
        video_metadata: Optional[dict] = None,
    ) -> dict:
        fps = self._get_video_fps(sample, video_metadata)
        frame_detections: dict = {}

        for item in items:
            frame_num = (
                int(self._timestamp_to_seconds(item.get("time", "00:00.00")) * fps) + 1
            )

            bbox = item.get("bbox_2d", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [max(0, min(1000, c)) for c in bbox[:4]]
            if x2 <= x1 or y2 <= y1:
                continue

            x, y, w, h = x1 / 1000, y1 / 1000, (x2 - x1) / 1000, (y2 - y1) / 1000
            label = item.get("text" if text_key else "label", "")
            detection = fol.Detection(label=label, bounding_box=[x, y, w, h])

            if text_key:
                detection[text_key] = item.get(text_key, "")

            if frame_num not in frame_detections:
                frame_detections[frame_num] = fol.Detections(detections=[])
            frame_detections[frame_num].detections.append(detection)

        return frame_detections

    def _merge_frame_labels(self, labels: dict, frame_detections: dict, key: str):
        for frame_num, dets in frame_detections.items():
            if frame_num not in labels:
                labels[frame_num] = {}
            labels[frame_num][key] = dets

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _timestamp_to_seconds(self, timestamp_str: str) -> float:
        """Convert 'mm:ss.ff' timestamp to total seconds."""
        match = re.match(r"(\d+):(\d+)\.(\d+)", str(timestamp_str))
        if not match:
            return 0.0
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        centiseconds = int(match.group(3))
        return minutes * 60 + seconds + centiseconds / 100.0

    def _get_video_fps(self, sample, video_metadata: Optional[dict] = None) -> float:
        """Get video FPS with fallback chain.

        Priority:
            1. video_metadata from processor
            2. sample.metadata.frame_rate
            3. decord fallback
        """
        if video_metadata and "fps" in video_metadata:
            return video_metadata["fps"]

        if sample is not None:
            meta = getattr(sample, "metadata", None)
            if meta is not None and hasattr(meta, "frame_rate"):
                return meta.frame_rate

            try:
                from decord import VideoReader, cpu
                vr = VideoReader(sample.filepath, ctx=cpu(0))
                return vr.get_avg_fps()
            except Exception:
                pass

        return 30.0

    def _extract_video_path(self, video) -> str:
        if isinstance(video, str):
            return video
        if hasattr(video, "inpath"):
            return video.inpath
        if hasattr(video, "path"):
            return video.path
        if hasattr(video, "filepath"):
            return video.filepath
        raise TypeError(f"Unsupported video type: {type(video)}")

    # -------------------------------------------------------------------------
    # predict / predict_all
    # -------------------------------------------------------------------------

    def predict(self, arg, sample=None):
        """Single-video inference.

        Accepts a filepath string, video reader object, or dict from GetItem.
        """
        if isinstance(arg, dict):
            batch_item = arg
        else:
            filepath = self._extract_video_path(arg)

            prompt = None
            if sample is not None and "prompt_field" in self._fields:
                field_name = self._fields["prompt_field"]
                if sample.has_field(field_name):
                    prompt = sample.get_field(field_name)

            metadata = getattr(sample, "metadata", None) if sample else None
            batch_item = {"filepath": filepath, "prompt": prompt, "metadata": metadata}

        results = self.predict_all([batch_item], samples=[sample] if sample else None)
        return results[0]

    def predict_all(self, batch: list, samples=None) -> list:
        """Batch video inference.

        Args:
            batch:   List of dicts with "filepath", "prompt", "metadata" keys
            samples: Optional list of FiftyOne samples

        Returns:
            List of label dicts, one per video
        """
        if not batch:
            return []

        if self._model is None:
            self._load_model()

        results = []
        for i, item in enumerate(batch):
            filepath = item["filepath"]
            metadata_from_batch = item.get("metadata")
            sample = samples[i] if samples else None

            needs_metadata = self.config.operation in (
                "comprehensive", "temporal_localization", "tracking", "ocr"
            )
            if needs_metadata and not metadata_from_batch:
                raise ValueError(
                    f"Operation '{self.config.operation}' requires sample metadata. "
                    "Call dataset.compute_metadata() first."
                )

            prompt_from_batch = item.get("prompt")
            if self.config.operation == "custom" and prompt_from_batch is not None:
                prompt = prompt_from_batch
            else:
                prompt = self.prompt

            messages = self._build_video_message(filepath, prompt)
            output_text, video_metadata = self._run_video_inference(messages)
            labels = self._parse_video_output(output_text, sample, video_metadata)
            results.append(labels)

        return results
