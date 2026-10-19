# Copyright (c) OpenMMLab. All rights reserved.
"""python examples/decision/decision.py CHECKPOINT [--image /trusted/local/image.png]"""
import argparse
import json

from lmdeploy import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--image', action='append', default=[])
    args = parser.parse_args()
    pipe = pipeline(args.checkpoint, task='intern-decision')
    try:
        result = pipe.decide(
            dict(state={'signal': 'red'},
                 images=args.image,
                 questions={
                     'action':
                     dict(type='choice',
                          instructions='Select an action.',
                          criteria={
                              'stop': 'Stop at the red light.',
                              'go': 'Drive through the red light.'
                          }),
                     'risk':
                     dict(type='score',
                          instructions='How risky is driving through the light?',
                          criteria={
                              '0': 'Safe',
                              '1': 'Some risk',
                              '2': 'Dangerous'
                          }),
                     'red':
                     dict(type='noul', instructions='Is the signal red?')
                 }))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        pipe.close()


if __name__ == '__main__':
    main()
