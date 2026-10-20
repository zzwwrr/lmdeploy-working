# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), src/service/uploads.py.
# Modified for LMDeploy: decode in memory, one loader for Python paths and HTTP uploads, and no EXIF rotation
# (the checkpoint's inference.py feeds pixels as stored).
"""Turn decision image inputs into validated, in-memory RGB images."""

from functools import partial
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lmdeploy.serve.core.decision_protocol import MAX_IMAGES
from lmdeploy.vl.media.image import ImageMediaIO

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TOTAL = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP', 'image/gif': 'GIF'}


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
    if content_type not in FORMATS:
        raise ValueError('Supported image formats: JPEG, PNG, WebP and static GIF.')
    payload = ImageMediaIO(strict_base64=True, max_bytes=MAX_UPLOAD_BYTES).decode_base64(content_type, encoded)
    return payload, FORMATS[content_type]


def _validate_image(source: Image.Image, expected_format: str | None) -> None:
    if expected_format is not None and source.format != expected_format:
        raise ValueError('Image content does not match its declared format.')
    if source.width * source.height > MAX_IMAGE_PIXELS:
        raise ValueError('Each image is limited to 16 million pixels.')
    if getattr(source, 'n_frames', 1) != 1:
        raise ValueError('Animated images are not supported; upload a still image.')


def load_images(items: list, *, allow_paths: bool, media_root: str | Path = '.') -> list[Image.Image]:
    """Load up to 8 images, in order.

    Upload objects (``{"data": ..., "type": ...}``) are always accepted. Plain strings are local file paths and are
    accepted only when ``allow_paths`` is set (trusted Python callers); remote URLs are never fetched.
    """
    if not isinstance(items, list) or len(items) > MAX_IMAGES:
        raise ValueError(f'images must be a list of at most {MAX_IMAGES} items.')
    images, total = [], 0
    try:
        for item in items:
            if isinstance(item, dict):
                payload, expected = _decode_upload(item)
                total += len(payload)
                if total > MAX_UPLOAD_TOTAL:
                    raise ValueError('Combined image upload is limited to 32 MB.')
                source = payload
            elif isinstance(item, (str, Path)) and str(item):
                if not allow_paths:
                    raise ValueError('Upload image bytes; server-side paths are not accepted.')
                if '://' in str(item):
                    raise ValueError('Remote image URLs are not supported; pass a local path or upload the bytes.')
                path = Path(item)
                source, expected = (path if path.is_absolute() else Path(media_root) / path), None
                if not source.is_file():
                    raise ValueError(f'Image file not found: {item}')
            else:
                raise ValueError('Each image must be a local path or an uploaded image object.')
            try:
                # Validation sees original headers/frames, before RGB conversion.
                # Keep the checkpoint's stored pixel orientation and drop metadata.
                loader = ImageMediaIO(validator=partial(_validate_image, expected_format=expected),
                                      clear_metadata=True)
                images.append(loader.load_bytes(source) if isinstance(source, bytes) else loader.load_file(source))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError('Invalid, incomplete or oversized image.') from exc
    except BaseException:
        for image in images:
            image.close()
        raise
    return images
