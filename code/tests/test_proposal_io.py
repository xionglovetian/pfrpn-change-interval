import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'proposal_io', ROOT / 'tools' / 'inference' / 'proposal_io.py')
proposal_io = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(proposal_io)


def candidate(candidate_id='tile_000123_000001'):
    return {
        'candidate_id': candidate_id,
        'bbox_xyxy': [120.0, 80.0, 310.0, 260.0],
        'objectness': 0.91,
        'proposal_score': 0.86,
        'level': 2,
        'uncertainty': 0.07,
    }


def meta():
    return {
        'num_queries': 900,
        'num_pseudo_tokens': 3,
        'sp_thr': 0.3,
        'sp_iter_num': 3,
        'topk': 2,
        'modalities_used': ['optical', 'sar', 'topography'],
    }


class ProposalIOTest(unittest.TestCase):

    def test_schema_is_valid(self):
        schema = json.loads(
            (ROOT / 'schemas' / 'proposal.v1.schema.json').read_text(
                encoding='utf-8'))
        Draft202012Validator.check_schema(schema)

    def test_builds_document_with_feature_ref(self):
        document = proposal_io.build_proposal_document(
            image_id='tile_000123',
            event_id='event_001',
            candidates=[candidate()],
            meta=meta(),
        )
        self.assertEqual(document['schema_version'], 'proposal.v1')
        self.assertEqual(
            document['candidates'][0]['roi_feature_ref'],
            'tile_000123.features.npz:tile_000123_000001',
        )
        proposal_io.validate_proposal_document(document)

    def test_writes_and_loads_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = proposal_io.write_proposal_artifacts(
                output_dir=tmp,
                split='test',
                event_id='event_001',
                image_id='tile_000123',
                candidates=[candidate()],
                features={'tile_000123_000001': np.ones((8, 7, 7))},
                meta=meta(),
            )
            loaded = proposal_io.load_proposal_document(paths['proposals'])
            self.assertEqual(len(loaded['candidates']), 1)
            with np.load(paths['features']) as archive:
                self.assertEqual(archive['tile_000123_000001'].shape, (8, 7, 7))

    def test_missing_feature_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(proposal_io.ProposalContractError):
                proposal_io.write_proposal_artifacts(
                    output_dir=tmp,
                    split='test',
                    event_id='event_001',
                    image_id='tile_000123',
                    candidates=[candidate()],
                    features={},
                    meta=meta(),
                )

    def test_duplicate_candidate_ids_are_rejected(self):
        with self.assertRaises(proposal_io.ProposalContractError):
            proposal_io.build_proposal_document(
                image_id='tile_000123',
                event_id='event_001',
                candidates=[candidate(), candidate()],
                meta=meta(),
            )

    def test_bad_bbox_is_rejected(self):
        bad = candidate()
        bad['bbox_xyxy'] = [10.0, 10.0, 10.0, 20.0]
        with self.assertRaises(proposal_io.ProposalContractError):
            proposal_io.build_proposal_document(
                image_id='tile_000123',
                event_id='event_001',
                candidates=[bad],
                meta=meta(),
            )

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(proposal_io.ProposalContractError):
            proposal_io.build_proposal_document(
                image_id='../escape',
                event_id='event_001',
                candidates=[candidate()],
                meta=meta(),
            )


if __name__ == '__main__':
    unittest.main()
