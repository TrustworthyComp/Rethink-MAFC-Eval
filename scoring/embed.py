#!/usr/bin/env python3
"""Generate evidence embeddings using a SentenceTransformer model."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_EMBEDDING_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
]

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def needs_query_prefix(model_name: str) -> bool:
    return "embeddinggemma" in model_name.lower()


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path.name}, got {type(data).__name__}")
    return data


def collect_evidence_texts(
    claims: List[Dict[str, Any]],
    use_query_prefix: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    sentences: List[str] = []
    positions: List[Tuple[int, int]] = []

    for claim_idx, claim in enumerate(claims):
        evidences = claim.get("evidences")
        if not isinstance(evidences, list):
            continue
        for ev_idx, ev in enumerate(evidences):
            text = ""
            if isinstance(ev, dict):
                text = str(ev.get("evidence", "")).strip()
            elif isinstance(ev, str):
                text = ev.strip()
            if not text:
                continue
            if use_query_prefix:
                text = f"task: fact checking | query: {text}"
            sentences.append(text)
            positions.append((claim_idx, ev_idx))

    return sentences, positions


def process_file(
    model: SentenceTransformer,
    model_name: str,
    input_path: Path,
    output_dir: Path,
) -> Optional[Tuple[Path, Path]]:
    """Process a single JSON file. Returns (json_path, npz_path) or None if skipped."""
    claims = load_json(input_path)
    use_prefix = needs_query_prefix(model_name)
    texts, positions = collect_evidence_texts(claims, use_query_prefix=use_prefix)

    if not texts:
        print(f"[Skip] {input_path.name}: no evidence text found.")
        return None

    npz_name = f"{input_path.stem}_embeddings.npz"
    npz_path = output_dir / npz_name
    t0 = time.time()

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False)

    emb_map: Dict[str, np.ndarray] = {}
    for emb_row, (claim_idx, ev_idx) in zip(embeddings, positions):
        key = f"claim{claim_idx}_e{ev_idx}"
        emb_map[key] = emb_row

        ev_obj = claims[claim_idx]["evidences"][ev_idx]
        if isinstance(ev_obj, dict):
            ev_obj["emb_key"] = key
        else:
            claims[claim_idx]["evidences"][ev_idx] = {"evidence": str(ev_obj), "emb_key": key}

    elapsed = time.time() - t0

    try:
        rel_npz = str(npz_path.relative_to(PROJECT_ROOT))
    except ValueError:
        rel_npz = str(npz_path)

    for claim in claims:
        if isinstance(claim.get("evidences"), list):
            claim["emb_path"] = rel_npz

    np.savez_compressed(npz_path, **emb_map)

    json_path = output_dir / f"{input_path.stem}_with_emb.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)

    print(f"[Done] {input_path.name} -> {json_path.name}, {npz_name}, evidence_count={len(texts)}, time={elapsed:.2f}s")
    return json_path, npz_path


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    json_path: Path
    npz_path: Path


def run(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    _loaded_model: Optional[SentenceTransformer] = None,
) -> List[EmbedResult]:
    """Generate embeddings. Returns list of (json, npz) paths per input file.

    Pass _loaded_model to reuse a SentenceTransformer across multiple calls.
    """
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = sorted(input_path.glob("*.json"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in: {input_path}")

    if _loaded_model is not None:
        model = _loaded_model
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Model: {model_name}")
        if needs_query_prefix(model_name):
            print("Using query prefix: 'task: fact checking | query: <evidence>'")
        model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    print(f"Input files: {len(input_files)}")
    results: List[EmbedResult] = []
    for json_path in input_files:
        pair = process_file(model, model_name, json_path, output_dir)
        if pair:
            results.append(EmbedResult(json_path=pair[0], npz_path=pair[1]))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence embeddings (NPZ + annotated JSON).")
    parser.add_argument("--input", required=True, help="Input JSON file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()
    run(Path(args.input), Path(args.output), args.model)


if __name__ == "__main__":
    main()
