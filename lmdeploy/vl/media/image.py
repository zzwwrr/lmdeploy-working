# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/image.py

import binascii
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import pybase64
from PIL import Image, ImageFile, UnidentifiedImageError

from .base import MediaIO

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_images(items: list,
                *,
                decode: Callable[[Any], tuple[bytes, str | None]],
                validate: Callable[[Image.Image, str | None], None] | None,
                max_images: int,
                max_total: int,
                allow_paths: bool,
                media_root: str | Path = '.') -> list[Image.Image]:
    """Load a request's images, in order.

    ``decode`` turns a mapping item into ``(payload, expected_format)``; it owns
    the upload wire format, MIME policy and per-image byte limit. Plain strings
    are local file paths and are read only when ``allow_paths`` is set (trusted
    callers); remote URLs are never fetched. ``validate`` sees each image before
    RGB conversion, so it can enforce format, frame count and pixel limits.

    The returned images are owned by the caller, which must close them.
    """
    if not isinstance(items, list) or len(items) > max_images:
        raise ValueError(f'images must be a list of at most {max_images} items.')
    images, total = [], 0
    try:
        for item in items:
            if isinstance(item, dict):
                payload, expected = decode(item)
                total += len(payload)
                if total > max_total:
                    raise ValueError(f'Combined image upload is limited to {max_total // (1024 * 1024)} MB.')
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
                loader = ImageMediaIO(validator=partial(validate, expected_format=expected) if validate else None,
                                      clear_metadata=True)
                images.append(loader.load_bytes(source) if isinstance(source, bytes) else loader.load_file(source))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError('Invalid, incomplete or oversized image.') from exc
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


class ImageMediaIO(MediaIO[Image.Image]):

    def __init__(self,
                 image_mode: str = 'RGB',
                 *,
                 validator: Callable[[Image.Image], None] | None = None,
                 clear_metadata: bool = False,
                 strict_base64: bool = False,
                 max_bytes: int | None = None,
                 **kwargs) -> None:
        super().__init__()
        if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1):
            raise ValueError('max_bytes must be a positive integer.')
        self.image_mode = image_mode
        self.validator = validator
        self.clear_metadata = clear_metadata
        self.strict_base64 = strict_base64
        self.max_bytes = max_bytes

        # for potential custom arguments from --media-io-kwargs
        self.kwargs = kwargs

    def _load_image(self, source: BytesIO | Path) -> Image.Image:
        # Validate the original format and frames before conversion loses that information.
        with Image.open(source) as opened:
            if self.validator is not None:
                self.validator(opened)
            image = opened.convert(self.image_mode)
        if self.clear_metadata:
            image.info.clear()
        return image

    def load_bytes(self, data: bytes) -> Image.Image:
        with BytesIO(data) as source:
            return self._load_image(source)

    def decode_base64(self, media_type: str, data: str) -> bytes:
        """Decode an upload, optionally limiting its size before allocation.

        ``media_type`` follows the media loader interface. Content-type checks
        belong to the caller's validator, which can inspect the original image.
        The byte limit applies to base64 uploads, not local files.
        """
        if self.max_bytes is not None and len(data) > 4 * ((self.max_bytes + 2) // 3):
            raise ValueError(f'Image upload exceeds {self.max_bytes} bytes.')
        try:
            payload = pybase64.b64decode(data, validate=self.strict_base64)
        except (ValueError, binascii.Error) as exc:
            raise ValueError('Invalid image base64.') from exc
        if (self.strict_base64 or self.max_bytes is not None) and not payload:
            raise ValueError('Image upload must not be empty.')
        if self.max_bytes is not None and len(payload) > self.max_bytes:
            raise ValueError(f'Image upload exceeds {self.max_bytes} bytes.')
        return payload

    def load_base64(self, media_type: str, data: str) -> Image.Image:
        return self.load_bytes(self.decode_base64(media_type, data))

    def load_file(self, file_path: Path) -> Image.Image:
        return self._load_image(file_path)

    def encode_base64(self, image: Image.Image, image_format: str = 'PNG') -> str:
        with BytesIO() as buffer:
            image = image.convert(self.image_mode)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode('utf-8')
