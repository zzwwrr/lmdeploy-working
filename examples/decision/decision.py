# Copyright (c) OpenMMLab. All rights reserved.
"""Score one Intern-Decision request, against a running api_server.

python examples/decision/decision.py --server http://127.0.0.1:23333 [--image image.png]
"""
import argparse
import base64
import json
import mimetypes

REQUEST = {
    'state': {
        'signal': 'red'
    },
    'questions': {
        'action': {
            'type': 'choice',
            'instructions': 'Select an action.',
            'criteria': {
                'stop': 'Stop at the red light.',
                'go': 'Drive through the red light.'
            }
        },
        'risk': {
            'type': 'score',
            'instructions': 'How risky is driving through the light?',
            'criteria': {
                '0': 'Safe',
                '1': 'Some risk',
                '2': 'Dangerous'
            }
        },
        'red': {
            'type': 'noul',
            'instructions': 'Is the signal red?'
        },
    },
}


def upload(path):
    """Upload a client-local image as bytes."""
    with open(path, 'rb') as f:
        return {'type': mimetypes.guess_type(path)[0], 'data': base64.b64encode(f.read()).decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--server', required=True, help='URL of `lmdeploy serve api_server ... --task intern-decision`')
    parser.add_argument('--image', action='append', default=[])
    args = parser.parse_args()

    import requests
    body = {**REQUEST, 'images': [upload(path) for path in args.image]}
    response = requests.post(f'{args.server}/v1/decisions', json=body, timeout=300)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
