# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility wrapper for the frozen proposal.v1 contract.

Canonical implementation: ``tools/inference/proposal_io.py``.
Authoritative schema: ``schemas/proposal.v1.schema.json``.
"""
import argparse
import json

from proposal_io import validate_proposal_document


def validate_proposal_dict(data):
    validate_proposal_document(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='proposal.v1 JSON file')
    args = parser.parse_args()
    with open(args.path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    validate_proposal_dict(data)
    print(f'OK: {args.path}')


if __name__ == '__main__':
    main()