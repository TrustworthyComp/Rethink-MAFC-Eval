#!/usr/bin/env python3
"""Compare evidence embeddings using cosine similarity + Hungarian matching."""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize

EVIDENCE_KEY_RE = re.compile(r"^claim(\d+)_e(\d+)$")


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path.name}, got {type(data).__name__}")
    return data


def parse_evidence_key(key: str) -> Optional[Tuple[int, int]]:
    m = EVIDENCE_KEY_RE.match(key)
    return (int(m.group(1)), int(m.group(2))) if m else None


def get_claim_text(obj: dict) -> str:
    text = obj.get("claim") or obj.get("claim_text")
    return str(text).strip() if text else ""


def build_claim_to_keys(claims: List[dict], npz_keys: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for key in npz_keys:
        parsed = parse_evidence_key(key)
        if parsed is None:
            continue
        idx, _ = parsed
        if idx < 0 or idx >= len(claims):
            continue
        ct = get_claim_text(claims[idx])
        if ct:
            mapping.setdefault(ct, []).append(key)
    return mapping


def gather_vectors(npz: np.lib.npyio.NpzFile, keys: List[str]) -> np.ndarray:
    vecs = [np.asarray(npz[k], dtype=np.float32) for k in sorted(keys)]
    return np.stack(vecs) if vecs else np.empty((0, 0), dtype=np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = np.linalg.norm(a, axis=1, keepdims=True)
    b_n = np.linalg.norm(b, axis=1, keepdims=True)
    a_n[a_n == 0] = 1e-12
    b_n[b_n == 0] = 1e-12
    return (a @ b.T) / (a_n @ b_n.T)


def hungarian_cosine(
    sim: np.ndarray,
    input_keys: List[str],
    ref_keys: List[str],
) -> Tuple[float, List[dict]]:
    if sim.size == 0:
        return 0.0, []
    rows, cols = scipy.optimize.linear_sum_assignment(sim, maximize=True)
    matched = []
    total = 0.0
    for r, c in zip(rows, cols):
        val = float(sim[r, c])
        total += val
        matched.append({"input_key": input_keys[r], "ref_key": ref_keys[c], "cosine": val})
    return total / float(sim.shape[1]), matched


def summarize(scores: List[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {"mean": None, "median": None, "max": None, "min": None}
    arr = np.asarray(scores, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "max": float(np.max(arr)), "min": float(np.min(arr))}


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

@dataclass
class EmbedMatchResult:
    output_path: Path
    summary: Dict[str, object]


def run(
    inp_npz: Path,
    inp_json: Path,
    ref_npz: Path,
    ref_json: Path,
    output_dir: Path,
) -> EmbedMatchResult:
    """Compare embeddings. Returns output path and summary dict."""
    inp_npz = inp_npz.expanduser().resolve()
    inp_json = inp_json.expanduser().resolve()
    ref_npz = ref_npz.expanduser().resolve()
    ref_json = ref_json.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in (inp_npz, inp_json, ref_npz, ref_json):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    inp_claims = load_json(inp_json)
    ref_claims = load_json(ref_json)
    inp_emb = np.load(inp_npz)
    ref_emb = np.load(ref_npz)

    inp_map = build_claim_to_keys(inp_claims, inp_emb.files)
    ref_map = build_claim_to_keys(ref_claims, ref_emb.files)
    common = sorted(set(inp_map) & set(ref_map))

    scores: List[float] = []
    details: List[dict] = []
    skipped = 0

    for ct in common:
        ik = sorted(inp_map[ct])
        rk = sorted(ref_map[ct])
        if not ik or not rk:
            skipped += 1
            continue
        iv = gather_vectors(inp_emb, ik)
        rv = gather_vectors(ref_emb, rk)
        if iv.size == 0 or rv.size == 0:
            skipped += 1
            continue

        sim = cosine_matrix(iv, rv)
        score, matched = hungarian_cosine(sim, ik, rk)
        scores.append(score)
        details.append({
            "score": score,
            "num_input": len(ik),
            "num_ref": len(rk),
            "matched_pairs": matched,
        })

    s = summarize(scores)

    print("=== Embedding cosine matching ===")
    print(f"input_npz: {inp_npz.name}")
    print(f"ref_npz: {ref_npz.name}")
    print(f"common_claims: {len(common)}")
    print(f"skipped: {skipped}")
    print(f"comparable: {len(scores)}")
    if s["mean"] is not None:
        print(f"score_mean: {s['mean']:.8f}")
    else:
        print("No comparable claims found.")

    result = {
        "summary": {
            "input_npz": inp_npz.name,
            "input_json": inp_json.name,
            "ref_npz": ref_npz.name,
            "ref_json": ref_json.name,
            "common_claims": len(common),
            "skipped": skipped,
            "comparable": len(scores),
            **{f"score_{k}": v for k, v in s.items()},
        },
        "details": details,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"embedding_scores_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"output: {out_path.name}")

    return EmbedMatchResult(output_path=out_path, summary=result["summary"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generated vs reference evidence embeddings (cosine + Hungarian)."
    )
    parser.add_argument("--npz", required=True, help="Input NPZ (generated evidence embeddings).")
    parser.add_argument("--json", required=True, help="Input *_with_emb.json (generated).")
    parser.add_argument("--ref-npz", required=True, help="Reference NPZ (reference evidence embeddings).")
    parser.add_argument("--ref-json", required=True, help="Reference *_with_emb.json.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    args = parser.parse_args()
    run(Path(args.npz), Path(args.json), Path(args.ref_npz), Path(args.ref_json), Path(args.output_dir))


if __name__ == "__main__":
    main()
