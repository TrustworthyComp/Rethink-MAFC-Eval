#!/usr/bin/env python3
"""Compute METEOR-based evidence similarity scores using Hungarian matching."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.optimize
import nltk
from nltk.tokenize import wordpunct_tokenize


def pairwise_meteor(candidate: str, reference: str) -> float:
    try:
        return nltk.translate.meteor_score.single_meteor_score(
            wordpunct_tokenize(reference),
            wordpunct_tokenize(candidate),
        )
    except LookupError as exc:
        raise LookupError(
            "NLTK resources missing. Run: python -m nltk.downloader wordnet omw-1.4"
        ) from exc


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path.name}, got {type(data).__name__}")
    return data


def get_claim_text(claim_obj: dict) -> str:
    text = claim_obj.get("claim") or claim_obj.get("claim_text")
    return str(text).strip() if text else ""


def get_evidence_texts(claim_obj: dict) -> List[str]:
    evidences = claim_obj.get("evidences")
    if not isinstance(evidences, list):
        return []
    out: List[str] = []
    for item in evidences:
        if isinstance(item, dict):
            text = str(item.get("evidence", "")).strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def build_claim_map(data: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for item in data:
        ct = get_claim_text(item)
        if ct and ct not in out:
            out[ct] = item
    return out


def meteor_matrix(gen_texts: List[str], ref_texts: List[str]) -> np.ndarray:
    mat = np.empty((len(gen_texts), len(ref_texts)), dtype=np.float64)
    for i, g in enumerate(gen_texts):
        for j, r in enumerate(ref_texts):
            mat[i, j] = pairwise_meteor(g, r)
    return mat


def hungarian_match(
    src_texts: List[str],
    tgt_texts: List[str],
) -> Tuple[float, List[dict]]:
    if not src_texts or not tgt_texts:
        return 0.0, []
    mat = meteor_matrix(src_texts, tgt_texts)
    if mat.size == 0:
        return 0.0, []
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(mat, maximize=True)
    matched: List[dict] = []
    total = 0.0
    for ri, ci in zip(row_idx, col_idx):
        s = float(mat[ri, ci])
        total += s
        matched.append({"gen_index": int(ri), "ref_index": int(ci), "meteor": s})
    score = total / float(mat.shape[1])
    return score, matched


def summarize(scores: List[float]) -> Dict[str, float]:
    arr = np.asarray(scores, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "median": float(np.median(arr)), "max": float(np.max(arr)), "min": float(np.min(arr))}


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

@dataclass
class MeteorResult:
    output_path: Path
    summary: Dict[str, object]


def run(gen_json: Path, ref_json: Path, output_dir: Path) -> MeteorResult:
    """Compute METEOR scores. Returns output path and summary dict."""
    gen_path = gen_json.expanduser().resolve()
    ref_path = ref_json.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in (gen_path, ref_path):
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    gen_data = load_json(gen_path)
    ref_data = load_json(ref_path)

    gen_map = build_claim_map(gen_data)
    ref_map = build_claim_map(ref_data)
    common_claims = sorted(set(gen_map) & set(ref_map))

    scores: List[float] = []
    details: List[dict] = []
    skipped = 0

    for claim_text in common_claims:
        gen_ev = get_evidence_texts(gen_map[claim_text])
        ref_ev = get_evidence_texts(ref_map[claim_text])

        score, matched = hungarian_match(gen_ev, ref_ev)
        if gen_ev and ref_ev:
            scores.append(score)
        else:
            skipped += 1

        details.append({
            "num_gen_evidences": len(gen_ev),
            "num_ref_evidences": len(ref_ev),
            "score": score,
            "matched_pairs": matched,
        })

    summary_stats = summarize(scores) if scores else {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}

    print("=== METEOR evidence matching ===")
    print(f"gen_json: {gen_path.name}")
    print(f"ref_json: {ref_path.name}")
    print(f"common_claims: {len(common_claims)}")
    print(f"skipped: {skipped}")
    print(f"comparable: {len(scores)}")
    print(f"score_mean: {summary_stats['mean']:.8f}")

    result = {
        "summary": {
            "gen_json": gen_path.name,
            "ref_json": ref_path.name,
            "common_claims": len(common_claims),
            "skipped_claims": skipped,
            "comparable_claims": len(scores),
            **{f"score_{k}": v for k, v in summary_stats.items()},
        },
        "details": details,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"meteor_scores_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"output: {out_path.name}")

    return MeteorResult(output_path=out_path, summary=result["summary"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare generated vs reference evidence using METEOR + Hungarian matching."
    )
    parser.add_argument("--gen-json", required=True, help="Generated extraction JSON.")
    parser.add_argument("--ref-json", required=True, help="Reference extraction JSON.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    args = parser.parse_args()
    run(Path(args.gen_json), Path(args.ref_json), Path(args.output_dir))


if __name__ == "__main__":
    main()
