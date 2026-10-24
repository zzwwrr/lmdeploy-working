# Copyright (c) OpenMMLab. All rights reserved.
"""Complete-input templates and image lifetime regressions, without model weights."""
import asyncio
import importlib
import threading

import pytest
from PIL import Image

from lmdeploy.model import BaseChatTemplate, HFChatTemplate
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


class _Tokenizer:

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        return ''.join(f"<{m['role']}>{m['content']}</end>" for m in messages)

    def encode(self, prompt, *, add_bos, add_special_tokens):
        assert add_bos is False
        assert add_special_tokens is False
        return [ord(c) for c in prompt]


def _assert_closed(image):
    with pytest.raises(ValueError, match='closed image'):
        image.getpixel((0, 0))


def test_complete_template_keeps_assistant_end_while_chat_trims_it():
    tokenizer = _Tokenizer()
    template = HFChatTemplate.__new__(HFChatTemplate)
    template.tokenizer = tokenizer
    template.assistant_end = '</end>'
    template.is_gpt_oss = False
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
    full = HFChatTemplate.render_complete(messages, tokenizer)
    assert template.messages2prompt(messages) == full[:-len('</end>')]


def test_scoring_uses_existing_tokenizer_without_bos_or_template_prefix_truncation():
    tokenizer = _Tokenizer()
    processor = MultimodalProcessor(tokenizer, BaseChatTemplate())
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': '<decision>'}]
    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True, scoring_input=True))
    expected = HFChatTemplate.render_complete(messages, tokenizer)
    assert result == {'prompt': expected, 'input_ids': [ord(c) for c in expected]}
    assert expected.endswith('<decision></end>')


def test_parallel_loader_failure_joins_other_workers_and_closes_partial_images(monkeypatch):

    async def run():
        entered, failed = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        created = []
        loop = asyncio.get_running_loop()

        def load(source, *args, **kwargs):
            if source == 'bad':
                loop.call_soon_threadsafe(failed.set)
                raise ValueError('broken image')
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            image = Image.new('RGB', (2, 3))
            created.append(image)
            return image

        monkeypatch.setattr(importlib.import_module('lmdeploy.serve.processors.multimodal'), 'load_from_url', load)
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': source}]} for source in ('slow', 'bad')]
        task = asyncio.create_task(MultimodalProcessor.async_parse_multimodal_item(messages))
        try:
            await asyncio.wait_for(asyncio.gather(entered.wait(), failed.wait()), 5)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match='broken image'):
            await task
        _assert_closed(created[0])

    asyncio.run(run())
