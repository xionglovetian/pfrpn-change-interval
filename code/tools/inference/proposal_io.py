"""Read/write helpers for the Module 1 ``proposal.v1`` artifact contract.

This module is intentionally independent from ``mmdet`` so it can be tested
without importing the detector package. The authoritative schema lives at
``schemas/proposal.v1.schema.json``.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 'proposal.v1'
DEFAULT_MODEL_VERSION = 'm2-pf-rpn-v1'
ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / 'schemas' / 'proposal.v1.schema.json'
_SAFE_COMPONENT = re.compile(r'^[A-Za-z0-9._-]+$')
_META_KEYS = (
    'num_queries', 'num_pseudo_tokens', 'sp_thr', 'sp_iter_num', 'topk',
    'modalities_used'
)
_CANDIDATE_KEYS = (
    'candidate_id', 'bbox_xyxy', 'objectness', 'proposal_score', 'level',
    'uncertainty'
)


class ProposalContractError(ValueError):
    """Raised when data cannot satisfy the proposal.v1 contract."""


def _plain(value: Any) -> Any:
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    return value


def _finite_number(value: Any, field: str) -> float:
    value = _plain(value)
    if isinstance(value, bool):
        raise ProposalContractError(f'{field} must be numeric, got bool')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProposalContractError(
            f'{field} must be numeric, got {value!r}') from exc
    if not math.isfinite(number):
        raise ProposalContractError(f'{field} must be finite, got {number!r}')
    return number


def _unit_interval(value: Any, field: str) -> float:
    number = _finite_number(value, field)
    if number < 0.0 or number > 1.0:
        raise ProposalContractError(
            f'{field} must be in [0, 1], got {number!r}')
    return number


def _nonnegative_int(value: Any, field: str) -> int:
    value = _plain(value)
    if isinstance(value, bool):
        raise ProposalContractError(f'{field} must be an integer')
    if isinstance(value, float) and not value.is_integer():
        raise ProposalContractError(f'{field} must be an integer')
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ProposalContractError(
            f'{field} must be an integer, got {value!r}') from exc
    if number < 0:
        raise ProposalContractError(
            f'{field} must be non-negative, got {number}')
    return number


def _safe_component(value: Any, field: str) -> str:
    text = str(value)
    if not text or text in {'.', '..'} or not _SAFE_COMPONENT.fullmatch(text):
        raise ProposalContractError(
            f'{field} must be a safe path component, got {text!r}')
    return text


def _bbox(value: Any) -> list[float]:
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ProposalContractError(
            'bbox_xyxy must contain exactly four numbers')
    bbox = [_finite_number(item, 'bbox_xyxy') for item in value]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ProposalContractError(
            f'bbox_xyxy is degenerate or has invalid ordering: {bbox}')
    return bbox


def _normalise_candidate(candidate: Mapping[str, Any], image_id: str) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ProposalContractError(
            f'candidate must be a mapping, got {type(candidate).__name__}')
    candidate_id = str(candidate.get('candidate_id', ''))
    if not candidate_id:
        raise ProposalContractError('candidate_id must be non-empty')
    missing = [key for key in _CANDIDATE_KEYS if key not in candidate]
    if missing:
        raise ProposalContractError(
            f'candidate {candidate_id!r} misses fields: {missing}')

    level = _nonnegative_int(candidate['level'], 'level')
    if level > 2:
        raise ProposalContractError(f'level must be 0, 1 or 2, got {level}')
    return {
        'candidate_id': candidate_id,
        'bbox_xyxy': _bbox(candidate['bbox_xyxy']),
        'objectness': _unit_interval(candidate['objectness'], 'objectness'),
        'proposal_score': _unit_interval(
            candidate['proposal_score'], 'proposal_score'),
        'level': level,
        'uncertainty': _unit_interval(
            candidate['uncertainty'], 'uncertainty'),
        'roi_feature_ref': f'{image_id}.features.npz:{candidate_id}',
    }


def _normalise_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(meta, Mapping):
        raise ProposalContractError('meta must be a mapping')
    missing = [key for key in _META_KEYS if key not in meta]
    if missing:
        raise ProposalContractError(f'meta misses fields: {missing}')
    modalities = _plain(meta['modalities_used'])
    if not isinstance(modalities, (list, tuple)) or not modalities:
        raise ProposalContractError('meta.modalities_used must be non-empty')
    modalities = [str(item) for item in modalities]
    if len(set(modalities)) != len(modalities):
        raise ProposalContractError('meta.modalities_used contains duplicates')
    return {
        'num_queries': _nonnegative_int(meta['num_queries'], 'num_queries'),
        'num_pseudo_tokens': _nonnegative_int(
            meta['num_pseudo_tokens'], 'num_pseudo_tokens'),
        'sp_thr': _unit_interval(meta['sp_thr'], 'sp_thr'),
        'sp_iter_num': _nonnegative_int(meta['sp_iter_num'], 'sp_iter_num'),
        'topk': _nonnegative_int(meta['topk'], 'topk'),
        'modalities_used': modalities,
    }


def build_proposal_document(
    *,
    image_id: str,
    event_id: str,
    candidates: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Any]:
    """Build and validate a JSON-serialisable proposal.v1 document."""
    image_id = _safe_component(image_id, 'image_id')
    event_id = _safe_component(event_id, 'event_id')
    if not str(model_version):
        raise ProposalContractError('model_version must be non-empty')
    normalised = [
        _normalise_candidate(candidate, image_id) for candidate in candidates
    ]
    candidate_ids = [item['candidate_id'] for item in normalised]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ProposalContractError('candidate_id values must be unique')
    document = {
        'schema_version': SCHEMA_VERSION,
        'image_id': image_id,
        'event_id': event_id,
        'model_version': str(model_version),
        'candidates': normalised,
        'meta': _normalise_meta(meta),
    }
    validate_proposal_document(document)
    return document


def validate_proposal_document(
    document: Mapping[str, Any],
    schema_path: Path | str = SCHEMA_PATH,
) -> None:
    """Validate a proposal document against the frozen JSON schema."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            'jsonschema is required to validate proposal.v1 artifacts') from exc
    schema = json.loads(Path(schema_path).read_text(encoding='utf-8'))
    try:
        jsonschema.Draft202012Validator(schema).validate(dict(document))
    except jsonschema.ValidationError as exc:
        path = '.'.join(str(part) for part in exc.absolute_path)
        location = path or '<root>'
        raise ProposalContractError(
            f'proposal.v1 validation failed at {location}: {exc.message}') from exc


def _normalise_features(
    features: Mapping[str, Any] | Sequence[Any],
    candidate_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    if isinstance(features, Mapping):
        keys = list(features)
        if set(keys) != set(candidate_ids):
            missing = sorted(set(candidate_ids) - set(keys))
            extra = sorted(set(keys) - set(candidate_ids))
            raise ProposalContractError(
                f'feature keys do not match candidates; missing={missing}, '
                f'extra={extra}')
        pairs = [(candidate_id, features[candidate_id])
                 for candidate_id in candidate_ids]
    else:
        if len(features) != len(candidate_ids):
            raise ProposalContractError(
                'feature sequence length must match candidate count')
        pairs = list(zip(candidate_ids, features))

    arrays: dict[str, np.ndarray] = {}
    for key, value in pairs:
        value = _plain(value)
        array = np.asarray(value)
        if array.ndim == 0:
            raise ProposalContractError(
                f'feature {key!r} must have at least one dimension')
        if not np.issubdtype(array.dtype, np.number):
            raise ProposalContractError(
                f'feature {key!r} must be numeric, got {array.dtype}')
        if not np.isfinite(array).all():
            raise ProposalContractError(
                f'feature {key!r} contains NaN or infinity')
        arrays[str(key)] = array
    return arrays


def write_proposal_artifacts(
    *,
    output_dir: Path | str,
    split: str,
    event_id: str,
    image_id: str,
    candidates: Sequence[Mapping[str, Any]],
    features: Mapping[str, Any] | Sequence[Any],
    meta: Mapping[str, Any],
    model_version: str = DEFAULT_MODEL_VERSION,
) -> dict[str, Path]:
    """Write proposal.v1 JSON and its referenced NPZ feature archive.

    Both inputs are validated before either file is written. Feature keys must
    exactly match candidate IDs; the order of a feature sequence is interpreted
    as the candidate order.
    """
    split = _safe_component(split, 'split')
    event_id = _safe_component(event_id, 'event_id')
    image_id = _safe_component(image_id, 'image_id')
    document = build_proposal_document(
        image_id=image_id,
        event_id=event_id,
        candidates=candidates,
        meta=meta,
        model_version=model_version,
    )
    candidate_ids = [item['candidate_id'] for item in document['candidates']]
    arrays = _normalise_features(features, candidate_ids)

    event_dir = Path(output_dir) / split / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    json_path = event_dir / f'{image_id}.proposals.json'
    npz_path = event_dir / f'{image_id}.features.npz'
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return {'proposals': json_path, 'features': npz_path}


def load_proposal_document(path: Path | str) -> dict[str, Any]:
    """Load and validate a proposal.v1 JSON file."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_proposal_document(document)
    return document


__all__ = [
    'ProposalContractError',
    'SCHEMA_PATH',
    'SCHEMA_VERSION',
    'build_proposal_document',
    'validate_proposal_document',
    'write_proposal_artifacts',
    'load_proposal_document',
]
