# Copyright (c) OpenMMLab. All rights reserved.
"""CPU preparation with request-scoped uploads and serialized processor access."""
import shutil
from threading import Lock

from .encoding import DecisionEncoder
from .multimodal import to_lmdeploy_multimodal
from .protocol import DecisionRequest, PreparedDecision
from .schema import _options
from .uploads import materialize_images


class DecisionInputProcessor:

    def __init__(self, checkpoint, hf_config, max_length=8192):
        self.encoder = DecisionEncoder(checkpoint, max_length=max_length)
        self.hf_config = hf_config
        self.lock = Lock()
        vision = hf_config.vision_config
        image_processor = self.encoder.processor.image_processor
        if (vision.spatial_merge_size != 2 or image_processor.merge_size != 2
                or image_processor.patch_size != vision.patch_size
                or image_processor.temporal_patch_size != vision.temporal_patch_size):
            raise ValueError('Decision processor and model vision parameters must match (merge_size=2)')
        vocab_size = hf_config.text_config.vocab_size
        if not 0 <= self.encoder.marker_id < vocab_size or any(not 0 <= value < vocab_size
                                                               for value in self.encoder._candidate_ids.values()):
            raise ValueError('Decision tokenizer IDs exceed the model vocabulary')

    def __call__(self, request, max_length=8192):
        request = DecisionRequest.model_validate(request)
        row = request.model_dump(exclude={'model', 'thinking'})
        directory = None
        try:
            if row['images'] and all(isinstance(item, dict) and 'data' in item for item in row['images']):
                row, directory = materialize_images(row)
            elif any(not isinstance(item, str) or '://' in item for item in row['images']):
                raise ValueError('Use local Python image paths or uploaded image objects; remote URLs are unsupported')
            with self.lock:
                encoded = self.encoder.encode(row)
                if encoded.input_length > min(8192, max_length):
                    raise ValueError(f'Input has {encoded.input_length} tokens, above capacity {min(8192, max_length)}')
                multimodal = to_lmdeploy_multimodal(encoded, self.hf_config)
            fields = encoded.compiled.fields
            options = tuple(_options(row['questions'][field]) for field in fields)
            return PreparedDecision(input_ids=encoded.batch['input_ids'][0].tolist(),
                                    multimodal=multimodal,
                                    read_positions=tuple(encoded.marker_positions.tolist()),
                                    candidate_ids=encoded.candidate_ids,
                                    fields=fields,
                                    option_values=tuple(tuple(value for value, _ in opts) for opts in options),
                                    question_types=tuple(row['questions'][field]['type'] for field in fields),
                                    legends=tuple(dict(opts) for opts in options),
                                    input_length=encoded.input_length)
        finally:
            if directory is not None:
                shutil.rmtree(directory)
