"""Qwen3.5-VL FiftyOne Zoo Model plugin.

Provides image and video understanding via Qwen3.5 vision-language model.

Image operations:  detect, point, classify, vqa, detect_3d
Video operations:  description, temporal_localization, tracking, ocr,
                   comprehensive, custom

Usage:
    import fiftyone.zoo as foz

    # Image model
    model = foz.load_zoo_model(
        "Qwen/Qwen3.5-9B",
        media_type="image",
        operation="detect",
        prompt="Detect all people in the image",
    )
    dataset.apply_model(model, label_field="detections")

    # Video model
    model = foz.load_zoo_model(
        "Qwen/Qwen3.5-9B",
        media_type="video",
        operation="description",
    )
    dataset.apply_model(model, label_field="description")
"""

import logging
import os

from huggingface_hub import snapshot_download
from fiftyone.operators import types

from .zoo import (
    IMAGE_OPERATIONS,
    VIDEO_OPERATIONS,
    Qwen35VLImageModel,
    Qwen35VLImageModelConfig,
    Qwen35VLVideoModel,
    Qwen35VLVideoModelConfig,
)

logger = logging.getLogger(__name__)


def download_model(model_name: str, model_path: str):
    """Download the Qwen3.5 model from HuggingFace.

    Args:
        model_name: HuggingFace repo ID (e.g. "Qwen/Qwen3.5-9B")
        model_path: Local directory to download into
    """
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(model_name: str = None, model_path: str = None, **kwargs):
    """Load a Qwen3.5-VL model for use with FiftyOne.

    Args:
        model_name: Model name (unused, kept for zoo compatibility)
        model_path: HuggingFace model ID or local path to model files.
            Defaults to "Qwen/Qwen3.5-9B".
        **kwargs: Config parameters. Key ones:

            media_type (str): "image" or "video". Default: "image".

            --- Image params ---
            operation (str):  One of detect, point, classify, vqa, detect_3d.
                              Default: "vqa".
            prompt (str):     User instruction for the image operation.
            system_prompt (str): Optional override for the default system prompt.
            camera_intrinsics_field (str): Sample field with {"fx","fy","cx","cy"}
                                           for detect_3d. Optional.
            fov (float):      Fallback field-of-view in degrees for pseudo camera
                              params in detect_3d. Default: 60.0.

            --- Video params ---
            operation (str):  One of description, temporal_localization, tracking,
                              ocr, comprehensive, custom. Default: "comprehensive".
            custom_prompt (str): Required when operation="custom".
            max_frames (int): Max frames to sample. Default: 120.
            sample_fps (int): Frame sampling rate. Default: 10.
            total_pixels (int): Quality/memory tradeoff. Default: 2048*32*32.

            --- Shared generation params ---
            max_new_tokens (int): Default 8192.
            do_sample (bool):     Default False.
            temperature (float):  Default 0.7.
            top_p (float):        Default 0.8.
            top_k (int):          Default 20.
            repetition_penalty (float): Default 1.0.

    Returns:
        Qwen35VLImageModel or Qwen35VLVideoModel
    """
    if model_path is None:
        model_path = "Qwen/Qwen3.5-9B"

    media_type = kwargs.pop("media_type", "image")

    config_dict = {"model_path": model_path}
    config_dict.update(kwargs)

    if media_type == "video":
        config = Qwen35VLVideoModelConfig(config_dict)
        return Qwen35VLVideoModel(config)

    config = Qwen35VLImageModelConfig(config_dict)
    return Qwen35VLImageModel(config)


def resolve_input(model_name: str, ctx):
    """Define FiftyOne operator UI inputs for this model.

    Args:
        model_name: The name of the model
        ctx:        An ExecutionContext

    Returns:
        fiftyone.operators.types.Property
    """
    inputs = types.Object()

    # -------------------------------------------------------------------------
    # Media type
    # -------------------------------------------------------------------------
    inputs.enum(
        "media_type",
        values=["image", "video"],
        default="image",
        label="Media Type",
        description="Whether to process images or videos",
    )

    # -------------------------------------------------------------------------
    # Operation (covers both image and video; validated at load time)
    # -------------------------------------------------------------------------
    all_operations = list(IMAGE_OPERATIONS.keys()) + [
        k for k in VIDEO_OPERATIONS.keys() if k != "custom"
    ]
    inputs.enum(
        "operation",
        values=all_operations,
        default="vqa",
        label="Operation",
        description=(
            "Image ops: detect, point, classify, vqa, detect_3d. "
            "Video ops: description, temporal_localization, tracking, ocr, comprehensive."
        ),
    )

    # -------------------------------------------------------------------------
    # Image-only parameters
    # -------------------------------------------------------------------------
    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Prompt",
        description="User instruction for image operations (detect, point, classify, vqa, detect_3d)",
    )

    inputs.str(
        "system_prompt",
        default=None,
        required=False,
        label="System Prompt Override",
        description="Optional: override the default system prompt for image operations",
    )

    inputs.str(
        "camera_intrinsics_field",
        default=None,
        required=False,
        label="Camera Intrinsics Field (detect_3d)",
        description=(
            "Sample field name containing camera intrinsics dict "
            "{'fx','fy','cx','cy'} for detect_3d. Leave empty to use FOV fallback."
        ),
    )

    inputs.int(
        "fov",
        default=60,
        label="Field of View in degrees (detect_3d fallback)",
        description="Used to generate pseudo camera params when no intrinsics field is set",
    )

    # -------------------------------------------------------------------------
    # Video-only parameters
    # -------------------------------------------------------------------------
    inputs.str(
        "custom_prompt",
        default=None,
        required=False,
        label="Custom Prompt (video only)",
        description="Required when operation is 'custom' for video",
    )

    inputs.int(
        "max_frames",
        default=120,
        label="Max Frames (video)",
        description="Maximum frames to sample from the video",
    )

    inputs.int(
        "sample_fps",
        default=10,
        label="Sample FPS (video)",
        description="Frame sampling rate for video processing",
    )

    inputs.int(
        "total_pixels",
        default=2048 * 32 * 32,
        label="Total Pixels (video)",
        description="Maximum total pixels for video quality/memory tradeoff",
    )

    # -------------------------------------------------------------------------
    # Shared generation parameters
    # -------------------------------------------------------------------------
    inputs.int(
        "max_new_tokens",
        default=8192,
        label="Max New Tokens",
        description="Maximum tokens to generate in the response",
    )

    inputs.bool(
        "do_sample",
        default=True,
        label="Use Sampling",
        description="Use sampling (True) vs greedy decoding (False)",
    )

    inputs.float(
        "temperature",
        default=0.7,
        label="Temperature",
        description="Sampling temperature (only used when Use Sampling is True)",
    )

    inputs.float(
        "top_p",
        default=0.95,
        label="Top-p",
        description="Nucleus sampling threshold (only used when Use Sampling is True)",
    )

    inputs.float(
        "min_p",
        default=0.0,
        label="Min-p",
        description=(
            "Minimum probability threshold for token sampling "
            "(only used when Use Sampling is True)"
        ),
    )

    inputs.float(
        "presence_penalty",
        default=1.5,
        label="Presence Penalty",
        description=(
            "Penalizes tokens that have already appeared in the output, "
            "reducing repetition. Applied regardless of sampling mode."
        ),
    )

    return types.Property(inputs)
