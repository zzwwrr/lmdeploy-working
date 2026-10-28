# Copyright (c) OpenMMLab. All rights reserved.
"""Complete-input template regressions, without model weights."""
import asyncio

from lmdeploy.model import BaseChatTemplate, HFChatTemplate
from lmdeploy.serve.processors.multimodal import MultimodalProcessor


def _render_complete(messages, provider):
    """Mirror the scoring branches' inline rendering call."""
    return provider.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class _Tokenizer:

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        return ''.join(f"<{m['role']}>{m['content']}</end>" for m in messages)

    def encode(self, prompt, *, add_bos, add_special_tokens):
        assert add_bos is False
        assert add_special_tokens is False
        return [ord(c) for c in prompt]


def test_complete_template_keeps_assistant_end_while_chat_trims_it():
    tokenizer = _Tokenizer()
    template = HFChatTemplate.__new__(HFChatTemplate)
    template.tokenizer = tokenizer
    template.assistant_end = '</end>'
    # Mirror the per-architecture flags set by HFChatTemplate.__init__ so the
    # test exercises a plain Qwen-style template.
    template.is_gpt_oss = False
    template.is_glm5_next = False
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
    full = _render_complete(messages, tokenizer)
    assert template.messages2prompt(messages) == full[:-len('</end>')]


def test_scoring_uses_existing_tokenizer_without_bos_or_template_prefix_truncation():
    tokenizer = _Tokenizer()
    processor = MultimodalProcessor(tokenizer, BaseChatTemplate())
    messages = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': '<decision>'}]
    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True, scoring_input=True))
    expected = _render_complete(messages, tokenizer)
    assert result == {'prompt': expected, 'input_ids': [ord(c) for c in expected]}
    assert expected.endswith('<decision></end>')
