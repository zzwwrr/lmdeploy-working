# Copyright (c) OpenMMLab. All rights reserved.
# Image upload policy adapted from Intern-Decision (0a8a2388), src/service/uploads.py.
# Copyright 2026 Shanghai AI Laboratory; SPDX-License-Identifier: Apache-2.0.
"""Upload policy for the decision protocol's image field."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from lmdeploy.serve.core.decision_protocol import MAX_IMAGES
from lmdeploy.vl.media.image import ImageMediaIO, load_images

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TOTAL = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP', 'image/gif': 'GIF'}


def _decode_upload(item: dict) -> tuple[bytes, str]:
    """Return the raw bytes and PIL format name of a base64 upload object."""
    data = item.get('data')
    if not isinstance(data, str):
        raise ValueError('Uploaded images need a base64 "data" string.')
    if len(data) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 128:
        raise ValueError('Each image is limited to 12 MB.')
    if data.startswith('data:'):
        header, separator, encoded = data.partition(',')
        if not separator or not header.endswith(';base64'):
            raise ValueError('Images must use base64 data URLs.')
        content_type = header[5:].split(';')[0].lower()
    else:
        encoded, content_type = data, str(item.get('type', '')).lower()
    declared = item.get('type')
    if declared is not None and str(declared).lower() != content_type:
        raise ValueError('Image MIME and data URL disagree')
    if content_type not in IMAGE_FORMATS:
        raise ValueError('Supported image formats: JPEG, PNG, WebP and static GIF.')
    payload = ImageMediaIO(strict_base64=True, max_bytes=MAX_UPLOAD_BYTES).decode_base64(content_type, encoded)
    return payload, IMAGE_FORMATS[content_type]


def _validate_image(source: Image.Image, expected_format: str | None) -> None:
    if expected_format is not None and source.format != expected_format:
        raise ValueError('Image content does not match its declared format.')
    if source.width * source.height > MAX_IMAGE_PIXELS:
        raise ValueError('Each image is limited to 16 million pixels.')
    if getattr(source, 'n_frames', 1) != 1:
        raise ValueError('Animated images are not supported; upload a still image.')


def load_decision_images(items: list,
                         *,
                         allow_paths: bool,
                         media_root: str | Path = '.') -> list[Image.Image]:
    """Load a decision request's images under the upload policy.

    Upload objects (``{"data": ..., "type": ...}``) are always accepted. Plain
    strings are local file paths and are accepted only when ``allow_paths`` is
    set (trusted Python callers); remote URLs are never fetched.
    """
    return load_images(items,
                       decode=_decode_upload,
                       validate=_validate_image,
                       max_images=MAX_IMAGES,
                       max_total=MAX_UPLOAD_TOTAL,
                       allow_paths=allow_paths,
                       media_root=media_root)
