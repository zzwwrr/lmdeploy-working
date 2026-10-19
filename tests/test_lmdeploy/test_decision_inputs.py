# Copyright 2026 Shanghai AI Laboratory
# SPDX-License-Identifier: Apache-2.0
# Adapted from Intern-Decision (0a8a2388), tests/test_lmdeploy_inputs.py.
# Modified for LMDeploy: standalone imports and opt-in checkpoint fixtures.
"""CPU bridge invariants and actual HF/LMDeploy MRoPE comparisons."""

import copy
import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from PIL import Image

from lmdeploy.serve.processors.intern_decision.encoding import DecisionEncoder, EncodedDecision
from lmdeploy.serve.processors.intern_decision.multimodal import to_lmdeploy_multimodal
from lmdeploy.serve.processors.intern_decision.schema import compile_row

CHECKPOINT = Path(os.environ.get('DECISION_TEST_CHECKPOINT', '/nonexistent-decision-checkpoint'))


def decision_row(images=None):
    return dict(state='红灯亮起，车辆应当等待。',
                images=images or [],
                questions={
                    'can_go': {
                        'type': 'noul',
                        'instructions': '现在可以通行吗？'
                    },
                    'color': {
                        'type': 'choice',
                        'criteria': {
                            'red': '红色',
                            'blue': '蓝色'
                        }
                    },
                    'score': {
                        'type': 'score',
                        'criteria': ['危险', '安全']
                    }
                })


def make_images(directory):
    paths = []
    for name, size, color in [('red.png', (64, 96), 'red'), ('blue.png', (160, 64), 'blue')]:
        path = Path(directory) / name
        with Image.new('RGB', size, color) as image:
            image.save(path)
        paths.append(str(path))
    return paths


def fixture():
    config = {
        "image_token_id": 100,
        "vision_start_token_id": 101,
        "vision_end_token_id": 102,
        "vision_config": {
            "spatial_merge_size": 2
        },
    }
    # First image: 8 patches / 2 tokens. Second: 16 patches / 4 tokens.
    ids = torch.tensor([[1, 101, 100, 100, 102, 2, 101, 100, 100, 100, 100, 102, 3]])
    batch = {
        "input_ids": ids,
        "image_grid_thw": torch.tensor([[1, 2, 4], [1, 4, 4]]),
        "pixel_values": torch.arange(72).reshape(24, 3).float(),
    }
    compiled = compile_row(decision_row(["first.png", "second.png"]), include_targets=False)
    return EncodedDecision(compiled, batch, torch.tensor([12]), ((4, 5), ), 13), config


class ImageBridgeTests(unittest.TestCase):

    def test_patch_and_token_offsets_have_different_units(self):
        encoded, config = fixture()
        result = to_lmdeploy_multimodal(encoded, config)
        self.assertEqual([item["offset"] for item in result], [2, 7])
        self.assertEqual([item["image_tokens"] for item in result], [2, 4])
        self.assertEqual([item["pixel_values"].shape for item in result], [(8, 3), (16, 3)])
        self.assertTrue(torch.equal(result[0]["pixel_values"], encoded.batch["pixel_values"][:8]))
        self.assertTrue(torch.equal(result[1]["pixel_values"], encoded.batch["pixel_values"][8:]))
        self.assertEqual(result[0]["image_grid_thw"].shape, (1, 3))
        self.assertEqual(result[1]["image_grid_thw"].tolist(), [[1, 4, 4]])
        object_config = SimpleNamespace(**(config | {"vision_config": SimpleNamespace(spatial_merge_size=2)}))
        self.assertEqual(to_lmdeploy_multimodal(encoded, object_config)[1]["offset"], 7)

    def test_plain_text_returns_none_but_stray_image_data_is_rejected(self):
        encoded, config = fixture()
        text = replace(encoded,
                       compiled=compile_row(decision_row()),
                       batch={"input_ids": torch.tensor([[1, 2, 3]])},
                       input_length=3)
        self.assertIsNone(to_lmdeploy_multimodal(text, config))
        for key in ("pixel_values", "image_grid_thw"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                to_lmdeploy_multimodal(replace(text, batch=text.batch | {key: encoded.batch[key]}), config)
        with self.assertRaises(ValueError):
            to_lmdeploy_multimodal(replace(text, batch={"input_ids": torch.tensor([[1, 100, 3]])}), config)

    def test_invalid_span_and_delimiter_locations(self):
        encoded, config = fixture()
        for position, value in ((1, 1), (4, 1), (3, 1), (5, 100), (0, 101), (12, 102)):
            batch = copy.deepcopy(encoded.batch)
            batch["input_ids"][0, position] = value
            with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                to_lmdeploy_multimodal(replace(encoded, batch=batch), config)
        with self.assertRaises(ValueError):
            to_lmdeploy_multimodal(replace(encoded, input_length=14), config)
        wrong_count = replace(encoded, compiled=compile_row(decision_row(["first.png"])))
        with self.assertRaises(ValueError):
            to_lmdeploy_multimodal(wrong_count, config)

    def test_invalid_grid_patch_count_and_nonfinite_pixels(self):
        encoded, config = fixture()
        invalid_grids = [
            torch.tensor([[2, 2, 4], [1, 4, 4]]),
            torch.tensor([[1, 3, 4], [1, 4, 4]]),
            torch.tensor([[1, 0, 4], [1, 4, 4]]),
            torch.tensor([[1, 2, 8], [1, 4, 4]]),
            torch.tensor([[1, 2, 4]]),
            torch.tensor([[1.0, 2.0, 4.0], [1.0, 4.0, 4.0]]),
            torch.tensor([1, 2, 4]),
        ]
        for grid in invalid_grids:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                to_lmdeploy_multimodal(replace(encoded, batch=encoded.batch | {"image_grid_thw": grid}), config)
        for pixels in (
                encoded.batch["pixel_values"][:-1],
                torch.cat([encoded.batch["pixel_values"], torch.zeros(1, 3)]),
                torch.full((24, 3), float("nan")),
                torch.full((24, 3), float("inf")),
                torch.ones(24, 3, dtype=torch.long),
                torch.ones(24),
        ):
            with self.subTest(shape=pixels.shape), self.assertRaises(ValueError):
                to_lmdeploy_multimodal(replace(encoded, batch=encoded.batch | {"pixel_values": pixels}), config)
        with self.assertRaisesRegex(ValueError, "spatial_merge_size=2"):
            to_lmdeploy_multimodal(encoded, config | {"vision_config": {"spatial_merge_size": 4}})
        self.assertFalse(torch.cuda.is_initialized())


def bind_methods(cls, names, **attrs):
    obj = SimpleNamespace(**attrs)
    for name in names:
        setattr(obj, name, MethodType(getattr(cls, name), obj))
    return obj


def context(start, length, image_data, metas=None):
    return SimpleNamespace(
        position_ids=torch.arange(start, start + length, dtype=torch.long)[None],
        q_seqlens=torch.tensor([length]),
        model_metas=metas,
        input_multimodals=[image_data] if image_data else None,
    )


@unittest.skipUnless((CHECKPOINT / "tokenizer.json").exists() and importlib.util.find_spec("lmdeploy"),
                     "Requires local model config and LMDeploy source")
class MRoPEComparisonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile

        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model as HFModel

        from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration as LMModel
        from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5InputProcessor

        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True)
        cls.encoder = DecisionEncoder(CHECKPOINT)
        paths = make_images(cls.temp.name)
        cls.items = [
            cls.encoder.encode(decision_row(images)) for images in (None, paths[:1], paths, paths[::-1], paths * 4)
        ]
        cls.hf = bind_methods(HFModel, ["get_vision_position_ids", "get_rope_index"], config=cls.config)
        cls.lm = bind_methods(LMModel, ["_get_model_metas", "_get_multimodal_pos_ids", "_update_model_meta_prefilling"])
        cls.processor = Qwen3_5InputProcessor(cls.config)
        print("MRoPE sources:", HFModel.__module__, LMModel.__module__)

    def compare(self, encoded, chunk_size=None):
        from lmdeploy.pytorch.engine.inputs_maker import LongContextChunker

        batch = encoded.batch
        ids = batch["input_ids"]
        raw = to_lmdeploy_multimodal(encoded, self.config)
        types = batch.get("mm_token_type_ids", (ids == self.config.image_token_id).long())
        hf_positions, hf_delta = self.hf.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=types,
            image_grid_thw=batch.get("image_grid_thw"),
            attention_mask=batch.get("attention_mask"),
        )
        image_data = None
        if raw:
            processed = self.processor.preprocess_input(ids[0].tolist(), raw)
            self.assertEqual(processed.input_ids, ids[0].tolist())
            image_data = processed.input_multimodals
        if chunk_size is None:
            ctx = context(0, encoded.input_length, image_data)
            metas = self.lm._update_model_meta_prefilling(ctx)
            lm_positions = ctx.mrope_position_ids
        else:
            seq = SimpleNamespace(num_history_ids=0, num_token_ids=encoded.input_length)
            seq.get_input_multimodals = lambda: copy.deepcopy(image_data or {})

            def set_step(step):
                seq.num_history_ids = step
                seq.num_token_ids = encoded.input_length - step

            seq.set_step = set_step
            chunker = LongContextChunker(chunk_size)
            chunker.set_seq(seq)
            outputs, metas = [], None
            for _ in range(encoded.input_length + 1):
                size, chunk_images = chunker.next_chunk_size()
                self.assertGreater(size, 0)
                ctx = context(seq.num_history_ids, size, chunk_images, metas)
                metas = self.lm._update_model_meta_prefilling(ctx)
                outputs.append(ctx.mrope_position_ids)
                if chunker.is_last_chunk():
                    self.assertEqual(seq.num_history_ids + size, encoded.input_length)
                    break
                chunker.update_step(SimpleNamespace(is_chunk=True, max_q_seqlen=size))
            else:
                self.fail("Chunker did not finish")
            lm_positions = torch.cat(outputs, dim=1)
        self.assertTrue(torch.equal(hf_positions[:, 0], lm_positions))
        self.assertEqual(int(hf_delta[0, 0]), int(metas[0]["mrope_delta"]))
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_hf_and_lmdeploy_positions_for_0_1_2_8_images(self):
        for index, item in enumerate(self.items):
            for chunk_size in (None, 64, 1024):
                with self.subTest(case=index, chunk_size=chunk_size):
                    self.compare(item, chunk_size)

    def test_image_crossing_prefill_boundary(self):
        item = self.items[2]
        first = int(torch.where(item.batch["input_ids"][0] == self.config.image_token_id)[0][0])
        # Place the first image token at 1020 so the real chunker must account
        # for the image spanning the usual 1024-token prefill boundary.
        prefix = 1020 - first
        self.assertGreater(prefix, 0)
        batch = dict(item.batch)
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if name not in batch:
                continue
            value = 123 if name == "input_ids" else int(name == "attention_mask")
            pad = torch.full((1, prefix), value, dtype=batch[name].dtype)
            batch[name] = torch.cat([pad, batch[name]], dim=1)
        item = replace(item,
                       batch=batch,
                       marker_positions=item.marker_positions + prefix,
                       input_length=item.input_length + prefix)
        self.compare(item, 1024)


if __name__ == "__main__":
    unittest.main()
