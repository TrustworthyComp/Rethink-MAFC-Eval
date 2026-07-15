#!/usr/bin/env python3
"""Extract evidence from fact-checking articles using LLM via OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

SUPPORTED_MODELS = [
    "gpt-5.2",
    "gpt-4o-mini-2024-07-18",
    "gemini-3.0-pro",
    "gemini-3.0-flash",
    "deepseek-v3.2",
    "qwen3.5-122b-a10b",
]

DATASET_PRESETS = {
    "averitec": str(PROJECT_ROOT / "data" / "input" / "averitec.json"),
    "claim2025q4": str(PROJECT_ROOT / "data" / "input" / "claim_review_2025q4.json"),
}

DEFAULT_GENERATION_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EXTRACT_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_ITEM_RETRIES = 3
DEFAULT_LLM_TIMEOUT = 120
DEFAULT_LLM_MAX_RETRIES = 3

# Modes and the item key holding the fact-checking article for each.
MODE_GENERATION = "generation"
MODE_REFERENCE = "reference"
GENERATION_KEY = "generation_text"
REVIEW_ARTICLE_KEY = "review_full_article"


class LLMParseError(Exception):
    """Raised when model output cannot be parsed as expected JSON."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class OpenRouterClient:
    """Minimal OpenRouter (OpenAI-compatible) HTTP client using stdlib."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.endpoint = f"{OPENROUTER_BASE_URL}/chat/completions"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                return parsed["choices"][0]["message"]["content"]
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    print(f"[HTTPError] status={exc.code} reason={exc.reason} attempt={attempt}/{self.max_retries}")
                elif isinstance(exc, urllib.error.URLError):
                    print(f"[URLError] reason={exc.reason} attempt={attempt}/{self.max_retries}")
                elif isinstance(exc, (TimeoutError, socket.timeout)):
                    print(f"[TimeoutError] attempt={attempt}/{self.max_retries}")
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(
            f"LLM request failed after {self.max_retries} retries "
            f"(last_error_type={type(last_error).__name__})."
        )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_yyyymmdd(raw: str) -> date:
    return datetime.strptime(raw, "%Y%m%d").date()


def _parse_iso_date(raw: str) -> Optional[date]:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _parse_ddmmyyyy(raw: str) -> Optional[date]:
    try:
        return datetime.strptime(raw, "%d-%m-%Y").date()
    except (TypeError, ValueError):
        return None


def get_date_published(item: Dict[str, Any]) -> Optional[date]:
    parsed = _parse_iso_date(item.get("date_published"))
    if parsed:
        return parsed
    reviews = item.get("reviews")
    if isinstance(reviews, list) and reviews and isinstance(reviews[0], dict):
        parsed = _parse_iso_date(reviews[0].get("date_published"))
        if parsed:
            return parsed
    parsed = _parse_ddmmyyyy(item.get("claim_date"))
    if parsed:
        return parsed
    return _parse_iso_date(item.get("date"))


# ---------------------------------------------------------------------------
# Claim / item helpers
# ---------------------------------------------------------------------------

def claim_to_str(claim: Any) -> str:
    if isinstance(claim, list):
        return " ".join(str(x).strip() for x in claim if str(x).strip()).strip()
    if claim is None:
        return ""
    return str(claim).strip()


def get_claim_field(item: Dict[str, Any]) -> Any:
    return item.get("claim")


def get_review_article(item: Dict[str, Any]) -> str:
    """Return the oracle fact-checking article (``review_full_article``), or ''."""
    value = item.get(REVIEW_ARTICLE_KEY)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def has_extracted(item: Dict[str, Any], mode: str = MODE_GENERATION) -> bool:
    """Whether an item already has valid extracted evidence for the given mode."""
    if not isinstance(item, dict):
        return False
    if "extract_evidence_reason" not in item or "evidences" not in item:
        return False
    # Generation mode additionally requires a non-empty generated article and reason.
    if mode == MODE_GENERATION:
        if not str(item.get(GENERATION_KEY, "")).strip():
            return False
        if not str(item.get("extract_evidence_reason", "")).strip():
            return False
    evidences = item.get("evidences")
    if not isinstance(evidences, list) or not evidences:
        return False
    return all(isinstance(ev, dict) and str(ev.get("evidence", "")).strip() for ev in evidences)


def build_item_signature(item: Dict[str, Any]) -> str:
    claim = claim_to_str(get_claim_field(item))
    review_url = str(item.get("review_url", "") or item.get("article", "")).strip()
    claim_date = str(item.get("claim_date", "") or item.get("date", "")).strip()
    published = ""
    reviews = item.get("reviews")
    if isinstance(reviews, list) and reviews and isinstance(reviews[0], dict):
        published = str(reviews[0].get("date_published", "")).strip()
    digest_src = f"{claim}\n{review_url}\n{claim_date}\n{published}"
    return hashlib.sha1(digest_src.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_block(text: str, require_dict: bool = True) -> Any:
    cleaned = _strip_code_fences(text)
    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(cleaned)
        if not require_dict or isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        candidate = block.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if not require_dict or isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[i:])
            if not require_dict or isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No valid JSON object found in model output", cleaned, 0)


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

def make_generation_messages(claim: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You write fact-checking articles."},
        {"role": "user", "content": f"Please write a fact-checking article to verify the claim.\nClaim: {claim}"},
    ]


def make_extraction_messages(claim: str, fact_checking_article: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a precise information extraction system.\n"
        "Your task is to extract evidence sentences from the fact-checking article "
        "that are directly related to the core factual content of claim.\n\n"
        "Rules:\n"
        "1. Only extract content that appears in the fact-checking article.\n"
        "2. Evidence must address the main factual assertion(s) made in claim.\n"
        "3. Do not infer, summarize, or add information.\n"
        "4. Do not extract sentences that merely restate the claim.\n"
        "5. Avoid duplication.\n\n"
        "Output strict JSON only:\n"
        '{"reason":"concise reasoning of extraction, <=50 words",'
        '"evidences":[{"evidence":"..."}]}'
    )
    user_payload = {"claim": claim, "fact_checking_article": fact_checking_article}
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def normalize_evidences(parsed: Any) -> List[Dict[str, str]]:
    if not isinstance(parsed, dict):
        return []
    raw_list = parsed.get("evidences", [])
    if not isinstance(raw_list, list):
        return []
    out: List[Dict[str, str]] = []
    seen: set = set()
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        evidence = str(item.get("evidence", "")).strip()
        if not evidence or evidence in seen:
            continue
        seen.add(evidence)
        out.append({"evidence": evidence})
    return out


def normalize_extraction_reason(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("reason", "")).strip()


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _extract_evidence(
    extract_client: OpenRouterClient,
    claim: str,
    fact_checking_article: str,
) -> Tuple[str, List[Dict[str, str]]]:
    """Run the extraction LLM call. Returns (reason, evidences)."""
    if not claim or not fact_checking_article:
        return "", []
    raw = extract_client.chat(make_extraction_messages(claim, fact_checking_article))
    try:
        extraction_json = extract_json_block(raw, require_dict=True)
    except Exception as exc:
        raise LLMParseError("extract_evidence", str(exc)) from exc
    return normalize_extraction_reason(extraction_json), normalize_evidences(extraction_json)


def _build_reference_item(
    item: Dict[str, Any],
    reason: str,
    evidences: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Preserve original key order, inserting new keys right after the article field."""
    new_item: Dict[str, Any] = {}
    inserted = False
    for key, value in item.items():
        new_item[key] = value
        if key == REVIEW_ARTICLE_KEY and not inserted:
            new_item["extract_evidence_reason"] = reason
            new_item["evidences"] = evidences
            inserted = True
    if not inserted:
        new_item["extract_evidence_reason"] = reason
        new_item["evidences"] = evidences
    return new_item


def process_item(
    item: Dict[str, Any],
    mode: str,
    extract_client: OpenRouterClient,
    gen_client: Optional[OpenRouterClient] = None,
) -> Dict[str, Any]:
    """Extract evidence for a single item according to ``mode``."""
    claim = claim_to_str(get_claim_field(item))

    if mode == MODE_REFERENCE:
        article = get_review_article(item)
        reason, evidences = _extract_evidence(extract_client, claim, article)
        return _build_reference_item(item, reason, evidences)

    # Generation mode: reuse an existing article or generate one from the claim.
    article = str(item.get(GENERATION_KEY, "")).strip()
    if not article and claim and gen_client is not None:
        article = gen_client.chat(make_generation_messages(claim)).strip()
    reason, evidences = _extract_evidence(extract_client, claim, article)

    new_item: Dict[str, Any] = dict(item)
    new_item[GENERATION_KEY] = article
    new_item["extract_evidence_reason"] = reason
    new_item["evidences"] = evidences
    return new_item


def filter_items(
    items: List[Dict[str, Any]],
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in items:
        if start_date and end_date:
            published = get_date_published(item)
            if not published or not (start_date <= published <= end_date):
                continue
        selected.append(item)
    selected.sort(key=lambda it: get_date_published(it) or date.max)
    return selected[:max_items]


def build_output_filename(
    name: str,
    start_date_raw: str,
    end_date_raw: str,
    item_count: int,
    ref: bool = False,
) -> str:
    date_part = f"_{start_date_raw}_{end_date_raw}" if (start_date_raw and end_date_raw) else ""
    suffix = "_ref_evidence" if ref else ""
    return f"{name}{suffix}{date_part}_{item_count}_items.json"


def process_single_item(
    task: Tuple[int, Dict[str, Any], str, str, str, str, int, int, int],
) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
    idx, item, api_key, gen_model, ext_model, mode, timeout, max_retries, item_retries = task
    ext_client = OpenRouterClient(api_key=api_key, model=ext_model, timeout=timeout, max_retries=max_retries)
    gen_client = (
        OpenRouterClient(api_key=api_key, model=gen_model, timeout=timeout, max_retries=max_retries)
        if mode == MODE_GENERATION
        else None
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, item_retries + 1):
        try:
            return idx, process_item(item, mode, ext_client, gen_client), None
        except Exception as exc:
            last_exc = exc
            if attempt < item_retries:
                time.sleep(0.8 * attempt)

    fallback = dict(item)
    if mode == MODE_GENERATION:
        fallback[GENERATION_KEY] = ""
    fallback["extract_evidence_reason"] = ""
    fallback["evidences"] = []
    claim = claim_to_str(get_claim_field(item))
    error_record: Dict[str, Any] = {
        "idx": idx,
        "error_type": type(last_exc).__name__ if last_exc else "unknown_error",
        "attempts": item_retries,
        "claim_signature": hashlib.sha1(claim.encode("utf-8")).hexdigest() if claim else "",
    }
    if isinstance(last_exc, LLMParseError):
        error_record["stage"] = last_exc.stage
    return idx, fallback, error_record


def append_error_record(error_path: Path, record: Dict[str, Any]) -> None:
    with error_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(output_path: Path, filtered_items: List[Dict[str, Any]], completed: Dict[int, Dict[str, Any]]) -> None:
    snapshot = [completed.get(i, item) for i, item in enumerate(filtered_items)]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def load_resume_cache(output_dir: Path, output_path: Path, prefix: str) -> Dict[str, Dict[str, Any]]:
    """Load already-extracted (generation-mode) items from sibling output JSONs for resume."""
    cache: Dict[str, Dict[str, Any]] = {}
    for path in output_dir.glob(f"{prefix}*.json"):
        if path == output_path:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            continue
        if not isinstance(existing, list):
            continue
        for out_item in existing:
            if isinstance(out_item, dict) and has_extracted(out_item, MODE_GENERATION):
                cache[build_item_signature(out_item)] = out_item
    return cache


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

@dataclass
class ExtractConfig:
    api_key: str
    dataset: str = "claim2025q4"
    input_file: str = ""
    output_dir: str = ""
    mode: str = MODE_GENERATION
    generation_model: str = DEFAULT_GENERATION_MODEL
    extract_model: str = DEFAULT_EXTRACT_MODEL
    start_date: str = ""
    end_date: str = ""
    max_items: int = 20
    workers: int = 4
    llm_timeout: int = DEFAULT_LLM_TIMEOUT
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    item_retries: int = DEFAULT_ITEM_RETRIES
    checkpoint_interval: int = 5
    no_resume: bool = False


def run(cfg: ExtractConfig) -> Path:
    """Run evidence extraction (generation or reference mode). Returns output JSON path."""
    if not cfg.api_key:
        raise ValueError("API key required.")
    if cfg.mode not in (MODE_GENERATION, MODE_REFERENCE):
        raise ValueError(f"Unknown mode '{cfg.mode}'. Choose '{MODE_GENERATION}' or '{MODE_REFERENCE}'.")
    if not cfg.input_file and cfg.dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'. Choose from {sorted(DATASET_PRESETS)} or pass input_file.")
    if bool(cfg.start_date) ^ bool(cfg.end_date):
        raise ValueError("start_date and end_date must be provided together.")

    is_reference = cfg.mode == MODE_REFERENCE

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    if cfg.start_date and cfg.end_date:
        start_date = _parse_yyyymmdd(cfg.start_date)
        end_date = _parse_yyyymmdd(cfg.end_date)
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date.")

    input_path = Path(cfg.input_file or DATASET_PRESETS[cfg.dataset]).expanduser().resolve()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON list.")

    filtered = filter_items(data, start_date, end_date, cfg.max_items)
    print(f"Selected {len(filtered)} items" + (f" in {cfg.start_date}..{cfg.end_date}" if start_date else "") + ".")

    if is_reference:
        with_article = sum(1 for it in filtered if get_review_article(it))
        if with_article == 0:
            print(f"[Warning] No selected item has a `{REVIEW_ARTICLE_KEY}` field; output will be empty evidence.")
        else:
            print(f"Items with {REVIEW_ARTICLE_KEY}: {with_article}/{len(filtered)}")

    # Reference output is named after the input stem; generation after the dataset.
    output_base = input_path.stem if is_reference else cfg.dataset
    output_name = build_output_filename(output_base, cfg.start_date, cfg.end_date, len(filtered), ref=is_reference)
    output_path = output_dir / output_name
    error_path = output_dir / f"{output_path.stem}.errors.jsonl"

    completed_by_idx: Dict[int, Dict[str, Any]] = {}
    sig_to_output: Dict[str, Dict[str, Any]] = {}
    sig_to_partial: Dict[str, Dict[str, Any]] = {}

    def _collect_resume(items: List[Any]) -> None:
        for out_item in items:
            if not isinstance(out_item, dict):
                continue
            sig = build_item_signature(out_item)
            if has_extracted(out_item, cfg.mode):
                sig_to_output[sig] = out_item
            elif not is_reference and str(out_item.get(GENERATION_KEY, "")).strip():
                sig_to_partial[sig] = out_item

    if not cfg.no_resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            _collect_resume(json.load(f))
    if not cfg.no_resume and not is_reference:
        sig_to_output.update(load_resume_cache(output_dir, output_path, f"{cfg.dataset}_"))

    pending: List[Tuple[int, Dict[str, Any], str, str, str, str, int, int, int]] = []
    reused = 0
    for i, item in enumerate(filtered):
        if has_extracted(item, cfg.mode):
            completed_by_idx[i] = item
            continue
        sig = build_item_signature(item)
        if sig in sig_to_output:
            completed_by_idx[i] = sig_to_output[sig]
            continue
        if not is_reference and sig in sig_to_partial and not str(item.get(GENERATION_KEY, "")).strip():
            item = dict(item)
            item[GENERATION_KEY] = sig_to_partial[sig][GENERATION_KEY]
            reused += 1
        pending.append((
            i, item, cfg.api_key, cfg.generation_model, cfg.extract_model, cfg.mode,
            cfg.llm_timeout, cfg.llm_max_retries, cfg.item_retries,
        ))

    total = len(filtered)
    skipped = len(completed_by_idx)
    print(f"Mode: {cfg.mode}")
    print(f"Total: {total}, skipped: {skipped}, pending: {len(pending)}" + (f", reused_generation: {reused}" if not is_reference else ""))
    if not is_reference:
        print(f"Generation model: {cfg.generation_model}")
    print(f"Extraction model: {cfg.extract_model}")

    desc = "Extract reference evidence" if is_reference else "Extract evidence"
    progress = tqdm(total=total, desc=desc) if tqdm else None
    if progress:
        progress.update(skipped)

    done_since_ckpt = 0
    if pending:
        workers = max(1, cfg.workers)

        async def _run() -> None:
            nonlocal done_since_ckpt
            sem = asyncio.Semaphore(workers)

            async def _one(t):
                async with sem:
                    return await asyncio.to_thread(process_single_item, t)

            tasks = [asyncio.create_task(_one(t)) for t in pending]
            for coro in asyncio.as_completed(tasks):
                idx, result, error = await coro
                if error:
                    print(f"  Failed idx={idx}: {error.get('error_type', 'unknown')}")
                    append_error_record(error_path, error)
                completed_by_idx[idx] = result
                done_since_ckpt += 1
                if progress:
                    progress.update(1)
                if cfg.checkpoint_interval > 0 and done_since_ckpt >= cfg.checkpoint_interval:
                    save_checkpoint(output_path, filtered, completed_by_idx)
                    done_since_ckpt = 0

        asyncio.run(_run())

    if progress:
        progress.close()

    output_items = [completed_by_idx.get(i, filtered[i]) for i in range(len(filtered))]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {output_path.name}")
    if error_path.exists():
        print(f"Errors: {error_path.name}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract evidence from fact-checking articles via OpenRouter LLM. "
                    "In 'generation' mode a fact-checking article is generated from the claim "
                    "first; in 'reference' mode evidence is extracted directly from the oracle "
                    "`review_full_article` field."
    )
    parser.add_argument("--mode", choices=[MODE_GENERATION, MODE_REFERENCE], default=MODE_GENERATION,
                        help="'generation': generate then extract. 'reference': extract from review_full_article.")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), default="claim2025q4", help="Dataset preset.")
    parser.add_argument("--input-file", default="", help="Explicit input JSON (overrides --dataset).")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL, help="Generation model.")
    parser.add_argument("--extract-model", default=DEFAULT_EXTRACT_MODEL, help="Extraction model.")
    parser.add_argument("--api-key", default="", help="OpenRouter API key (or set OPENROUTER_API_KEY).")
    parser.add_argument("--start-date", default="", help="Filter start date (YYYYMMDD).")
    parser.add_argument("--end-date", default="", help="Filter end date (YYYYMMDD).")
    parser.add_argument("--max-items", type=int, default=20, help="Max claims to process.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM workers.")
    parser.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT, help="HTTP timeout (seconds).")
    parser.add_argument("--llm-max-retries", type=int, default=DEFAULT_LLM_MAX_RETRIES, help="Retries per LLM call.")
    parser.add_argument("--item-retries", type=int, default=DEFAULT_ITEM_RETRIES, help="Retries per item.")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Checkpoint every N items.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing output.")
    args = parser.parse_args()

    cfg = ExtractConfig(
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY", ""),
        dataset=args.dataset,
        input_file=args.input_file,
        output_dir=args.output_dir,
        mode=args.mode,
        generation_model=args.generation_model,
        extract_model=args.extract_model,
        start_date=args.start_date,
        end_date=args.end_date,
        max_items=args.max_items,
        workers=args.workers,
        llm_timeout=args.llm_timeout,
        llm_max_retries=args.llm_max_retries,
        item_retries=args.item_retries,
        checkpoint_interval=args.checkpoint_interval,
        no_resume=args.no_resume,
    )
    run(cfg)


if __name__ == "__main__":
    main()
