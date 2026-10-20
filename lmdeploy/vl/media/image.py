# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/media/image.py

import binascii
from io import BytesIO
from pathlib import Path
from typing import Callable

import pybase64
from PIL import Image, ImageFile

from .base import MediaIO

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
