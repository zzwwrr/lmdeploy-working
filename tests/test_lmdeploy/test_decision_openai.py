# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the production wire models and parser with fixed engine scores."""
import copy
import math
from types import SimpleNamespace

import pytest
from test_decision import MARKER, config

from lmdeploy.intern_decision_encoding import compile_row
from lmdeploy.serve.core.async_engine import GenOut
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.decision_protocol import OpenAIDecisionRequest
from lmdeploy.serve.parsers.decision_parser import DecisionParser


def request(questions=None, **kwargs):
    return dict(model='intern-decision', input='The export failed.', questions=questions or [
        dict(type='predicate', name='blocked', instructions='Is work blocked?'),
        dict(type='choice', name='team', instructions='Who handles this?', choices=[
            dict(value='support', description='Help'), dict(value='billing', description='Payments')]),
        dict(type='score', name='severity', instructions='How severe?', levels=[
            dict(label='Low'), dict(label='Medium', description='A workaround exists'), dict(label='High')]),
    ], **kwargs)


def prepared_output(parser, probabilities):
    """Substitute only tokenizer/engine work; production compilation and row selection run."""
    inputs = parser.prepare_input()
    prepared = SimpleNamespace(inputs={'input_ids': [1] + [MARKER, 3] * len(probabilities)},
                               gen_config=inputs.gen_config)
    parser.finalize_input(prepared)
    rows = []
    for distribution in probabilities:
        rows.append({32 + i: math.log(p) for i, p in enumerate(distribution)})
        rows.append({32: -1000., 33: 1000., 34: -1000.})
    return GenOut(response='', input_token_len=37, generate_token_len=0, cached_tokens=11,
                  logprobs=rows, finish_reason='length')


def test_three_types_use_existing_compiler_and_decoder():
    body = request()
    original = copy.deepcopy(body)
    parser = DecisionParser(body, config(temperature=1.))
    legacy = dict(state=body['input'], questions={
        'blocked': dict(type='noul', instructions='Is work blocked?'),
        'team': dict(type='choice', instructions='Who handles this?',
                     criteria={'support': 'Help', 'billing': 'Payments'}),
        'severity': dict(type='score', instructions='How severe?',
                         criteria=['Low', 'Medium: A workaround exists', 'High']),
    })
    assert parser.prepare_input().messages == compile_row(legacy).messages
    output = prepared_output(parser, [[.8, .2], [.7, .3], [.2, .3, .5]])
    result = parser.parse_output(output, model_name='intern-decision', timing={})
    assert body == original
    predicate, choice, score = result['answers']
    assert [a['name'] for a in result['answers']] == ['blocked', 'team', 'severity']
    assert predicate['probability'] == pytest.approx(.2)  # P(yes), not max probability (.8).
    assert choice['choice'] == 'support' and choice['confidence'] == pytest.approx(.7)
    assert [p['value'] for p in choice['probabilities']] == ['support', 'billing']
    assert score['score'] == pytest.approx(1.3)
    assert [p['value'] for p in score['probabilities']] == [0, 1, 2]
    assert [p['label'] for p in score['probabilities']] == ['Low', 'Medium', 'High']
    assert result['usage'] == dict(input_tokens=37, output_tokens=0, total_tokens=37,
                                  input_tokens_details={'cached_tokens': 11, 'cache_write_tokens': 0},
                                  output_tokens_details={'reasoning_tokens': 0})
    native = DecisionParser(legacy, config(temperature=1.))
    native_result = native.parse_output(prepared_output(native, [[.8, .2], [.7, .3], [.2, .3, .5]]),
                                        model_name='intern-decision', timing={})
    assert native_result['usage']['output_tokens'] == 3
    assert result['answers'][0]['probability'] == native_result['answers']['blocked']['noul']


def test_typed_choices_and_interleaved_parsers():
    typed = request([dict(type='choice', name='typed', instructions='Choose',
                          choices=[dict(value=True), dict(value='true')])])
    other = request([dict(type='choice', name='other', instructions='Choose',
                          choices=[dict(value='first'), dict(value='second')])])
    first, second = (DecisionParser(body, config(temperature=1.)) for body in (typed, other))
    first_output = prepared_output(first, [[.8, .2]])
    second_output = prepared_output(second, [[.1, .9]])
    assert 'A = true:' in first.compiled.messages[1]['content']
    assert 'B = "true":' in first.compiled.messages[1]['content']
    second_result = second.parse_output(second_output, model_name='intern-decision', timing={})
    assert second_result['answers'][0]['choice'] == 'second'
    answer = first.parse_output(first_output, model_name='intern-decision', timing={})['answers'][0]
    assert answer['name'] == 'typed' and answer['choice'] is True
    assert [p['value'] for p in answer['probabilities']] == [True, 'true']
    # Reverse the winner through the real decoder, not a hand-built internal answer.
    answer = first.parse_output(prepared_output(first, [[.2, .8]]),
                                model_name='intern-decision', timing={})['answers'][0]
    assert answer['choice'] == 'true' and type(answer['choice']) is str


def test_names_preserve_order_without_collisions():
    names = [None, 'question_0', 'same', 'same', '', '_question_0']
    body = request([dict(type='predicate', instructions='Check', **({'name': name} if name is not None else {}))
                    for name in names])
    parser = DecisionParser(body, config(temperature=1.))
    distributions = [[1 - p, p] for p in (.1, .2, .3, .4, .5, .6)]
    answers = parser.parse_output(prepared_output(parser, distributions),
                                  model_name='intern-decision', timing={})['answers']
    assert len(parser.compiled.fields) == len(set(parser.compiled.fields)) == len(names)
    assert [a['name'] for a in answers] == names
    assert [a['probability'] for a in answers] == pytest.approx([.1, .2, .3, .4, .5, .6])
    assert 'question_0' in parser.compiled.fields and '_question_0' in parser.compiled.fields


@pytest.mark.parametrize('change', [
    {'input': []},
    {'input': 1}, {'model': None}, {'state': 'mixed protocols'}, {'images': []},
    {'prompt_cache_key': 'unsupported'}, {'questions': []},
    {'questions': [dict(type='predicate', instructions='x')] * 17},
    {'questions': [dict(type='predicate', instructions=[{'type': 'input_text', 'text': 'x'}])]},
    {'questions': [dict(type='predicate', instructions='x', unexpected=True)]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 'a'}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 'a'}, {'value': 'a'}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': 1}, {'value': 2}])]},
    {'questions': [dict(type='choice', instructions='x', choices=[{'value': str(i)} for i in range(63)])]},
    {'questions': [dict(type='score', instructions='x', levels=[{'label': 'one'}])]},
    {'questions': [dict(type='score', instructions='x', levels=[{'label': str(i)} for i in range(11)])]},
])
def test_reject_unsupported_input(change):
    with pytest.raises(ValueError):
        OpenAIDecisionRequest.model_validate({**request(), **change})


@pytest.mark.parametrize('value', [
    [{'role': 'assistant', 'content': 'x'}],
    [{'role': 'user', 'content': []}],
    [{'role': 'user', 'content': [{'type': 'input_audio', 'data': 'x'}]}],
    *[[{'role': 'user', 'content': [dict(type='input_image', **image)]}] for image in [
        {'image_url': '/server/private.png'}, {'image_url': 'file:///image.png'}, {'image_url': 'https://'},
        {'image_url': 'not-base64'}, {'image_url': 'https://example.com/a.png', 'detail': 'high'},
    ]],
])
def test_reject_unsupported_message_content(value):
    with pytest.raises(ValueError):
        OpenAIDecisionRequest.model_validate({**request(), 'input': value})


def test_single_string_message_preserves_text_prompt():
    body = request()
    plain = DecisionParser(body, config()).prepare_input()
    body['input'] = [dict(type='message', role='user', content=body['input'])]
    assert DecisionParser(body, config()).prepare_input().messages == plain.messages


def test_reserved_marker_and_missing_scores():
    parser = DecisionParser({**request(), 'input': '<decision>'}, config())
    with pytest.raises(ValueError, match='Reserved'):
        parser.prepare_input()
    parser = DecisionParser(request(), config())
    output = prepared_output(parser, [[.8, .2], [.7, .3], [.2, .3, .5]])
    del output.logprobs[0][33]
    with pytest.raises(RequestError, match='lack a requested candidate'):
        parser.parse_output(output, model_name='intern-decision', timing={})
