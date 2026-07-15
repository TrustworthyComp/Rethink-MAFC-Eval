#!/usr/bin/env python3
"""Unified pipeline: extract evidence -> score with chosen similarity model."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from extraction.extract import (
    DATASET_PRESETS,
    ExtractConfig,
    _parse_yyyymmdd,
    build_output_filename,
    filter_items,
    run as extract_evidence,
)
from scoring.meteor import run as score_meteor
from scoring.embed import run as generate_embeddings, EmbedResult
from scoring.embed_match import run as match_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent

SUPPORTED_LLM_MODELS = [
    "gpt-5.2",
    "gpt-4o-mini-2024-07-18",
    "gemini-3.0-pro",
    "gemini-3.0-flash",
    "deepseek-v3.2",
    "qwen3.5-122b-a10b",
]

SIMILARITY_MODELS = [
    "meteor",
    "Qwen/Qwen3-Embedding-0.6B",
    "google/embeddinggemma-300m",
]


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()) or "unknown"


def expected_extraction_name(dataset: str, start_date: str, end_date: str, max_items: int) -> str:
    """Compute the extraction filename `extract.py` would produce for these inputs.

    The name encodes dataset + date range + item count, so a filename match implies
    the same set of items (consistency guard for reuse).
    """
    with Path(DATASET_PRESETS[dataset]).open("r", encoding="utf-8") as f:
        data = json.load(f)
    sd: Optional[date] = None
    ed: Optional[date] = None
    if start_date and end_date:
        sd = _parse_yyyymmdd(start_date)
        ed = _parse_yyyymmdd(end_date)
    filtered = filter_items(data, sd, ed, max_items)
    return build_output_filename(dataset, start_date, end_date, len(filtered))


def find_reusable_extraction(
    output_base: Path,
    gen_part: str,
    ds_part: str,
    expected_name: str,
) -> Optional[Path]:
    """Search sibling `output/{gen}-{ds}-*/extraction/{expected_name}` for a reusable file.

    Only the similarity-model suffix differs across siblings, so any match was produced
    by the same generation model on the same item set. Returns the first valid one.
    """
    if not output_base.exists():
        return None
    for run_dir in sorted(output_base.glob(f"{gen_part}-{ds_part}-*")):
        candidate = run_dir / "extraction" / expected_name
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue
        if isinstance(items, list) and items:
            return candidate
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One-command pipeline: extract evidence, then score with METEOR or embedding similarity."
    )
    p.add_argument("--generation-model", required=True, help="LLM model for fact-check generation.")
    p.add_argument("--extract-model", default="", help="LLM model for extraction (default: same as generation).")
    p.add_argument("--dataset", choices=["averitec", "claim2025q4"], default="claim2025q4", help="Dataset.")
    p.add_argument("--similarity-model", choices=SIMILARITY_MODELS, required=True, help="Similarity scoring model.")
    p.add_argument("--ref-json", required=True, help="Reference extraction JSON (user-generated).")
    p.add_argument("--gen-json", default="",
                   help="Reuse a specific generation extraction JSON and skip Step 1 "
                        "(e.g. when re-scoring with a different --similarity-model).")
    p.add_argument("--force-extraction", action="store_true",
                   help="Always run Step 1, even if a matching extraction from another "
                        "similarity-model run could be reused.")
    p.add_argument("--api-key", default="", help="OpenRouter API key (or set OPENROUTER_API_KEY).")
    p.add_argument("--start-date", default="", help="Filter start date (YYYYMMDD).")
    p.add_argument("--end-date", default="", help="Filter end date (YYYYMMDD).")
    p.add_argument("--max-items", type=int, default=20, help="Max claims.")
    p.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers.")
    p.add_argument("--force-embedding", action="store_true", help="Regenerate embeddings even if they exist.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    extract_model = args.extract_model or args.generation_model
    ref_json_path = Path(args.ref_json).expanduser().resolve()
    if not ref_json_path.exists():
        raise FileNotFoundError(f"Reference JSON not found: {ref_json_path}")

    gen_part = sanitize(args.generation_model)
    ds_part = sanitize(args.dataset)
    sim_part = sanitize(args.similarity_model)
    output_root = PROJECT_ROOT / "output" / f"{gen_part}-{ds_part}-{sim_part}"
    extract_dir = output_root / "extraction"
    score_dir = output_root / "scores"
    embed_dir = output_root / "embeddings"

    # ── Decide the Step 1 source: explicit reuse > auto-reuse > fresh extraction ──
    reuse_source: Optional[Path] = None
    reuse_reason = ""
    if args.gen_json:
        reuse_source = Path(args.gen_json).expanduser().resolve()
        if not reuse_source.exists():
            raise FileNotFoundError(f"--gen-json not found: {reuse_source}")
        reuse_reason = "--gen-json"
    elif not args.force_extraction:
        try:
            expected = expected_extraction_name(args.dataset, args.start_date, args.end_date, args.max_items)
            found = find_reusable_extraction(PROJECT_ROOT / "output", gen_part, ds_part, expected)
            if found is not None:
                reuse_source = found
                reuse_reason = "auto-detected (same generation model + item set)"
        except Exception as exc:
            print(f"[reuse] auto-detect skipped: {exc}")

    # API key is only needed for Step 1 (extraction). Skip the check when reusing extraction.
    if reuse_source is None and not api_key:
        raise ValueError("API key required. Set OPENROUTER_API_KEY or pass --api-key.")

    for d in (extract_dir, score_dir, embed_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"output: {output_root}")

    # ── Step 1: Extract evidence ──────────────────────────────────────────
    if reuse_source is not None:
        print(f"\n==== Step 1: Extract evidence (skipped, reusing {reuse_reason}) ====")
        print(f"gen_json: {reuse_source}")
        gen_json = reuse_source
    else:
        print("\n==== Step 1: Extract evidence ====")
        gen_json = extract_evidence(ExtractConfig(
            api_key=api_key,
            dataset=args.dataset,
            output_dir=str(extract_dir),
            generation_model=args.generation_model,
            extract_model=extract_model,
            start_date=args.start_date,
            end_date=args.end_date,
            max_items=args.max_items,
            workers=args.workers,
        ))

    if args.similarity_model == "meteor":
        # ── Step 2: METEOR scoring ────────────────────────────────────────
        print("\n==== Step 2: METEOR scoring ====")
        result = score_meteor(gen_json, ref_json_path, score_dir)
        print(f"\n==== Result ====")
        print(f"METEOR score_mean: {result.summary.get('score_mean')}")

    else:
        # ── Step 2: Generate embeddings (generated) ───────────────────────
        emb_json = embed_dir / f"{gen_json.stem}_with_emb.json"
        emb_npz = embed_dir / f"{gen_json.stem}_embeddings.npz"

        if emb_json.exists() and emb_npz.exists() and not args.force_embedding:
            print("\n==== Step 2: Generate embeddings (skipped, files exist) ====")
            gen_emb = EmbedResult(json_path=emb_json, npz_path=emb_npz)
        else:
            print("\n==== Step 2: Generate embeddings (generated) ====")
            gen_results = generate_embeddings(gen_json, embed_dir, args.similarity_model)
            if not gen_results:
                raise RuntimeError("Embedding generation produced no output.")
            gen_emb = gen_results[0]

        # ── Step 3: Generate embeddings (reference) ───────────────────────
        ref_emb_json = embed_dir / f"{ref_json_path.stem}_with_emb.json"
        ref_emb_npz = embed_dir / f"{ref_json_path.stem}_embeddings.npz"

        if ref_emb_json.exists() and ref_emb_npz.exists() and not args.force_embedding:
            print("\n==== Step 3: Generate embeddings (reference, skipped) ====")
            ref_emb = EmbedResult(json_path=ref_emb_json, npz_path=ref_emb_npz)
        else:
            print("\n==== Step 3: Generate embeddings (reference) ====")
            ref_results = generate_embeddings(ref_json_path, embed_dir, args.similarity_model)
            if not ref_results:
                raise RuntimeError("Reference embedding generation produced no output.")
            ref_emb = ref_results[0]

        # ── Step 4: Embedding cosine scoring ──────────────────────────────
        print("\n==== Step 4: Embedding cosine scoring ====")
        result = match_embeddings(
            gen_emb.npz_path, gen_emb.json_path,
            ref_emb.npz_path, ref_emb.json_path,
            score_dir,
        )
        print(f"\n==== Result ====")
        print(f"Embedding score_mean: {result.summary.get('score_mean')}")


if __name__ == "__main__":
    main()
