# [MM-2026] Novel Claim or Déjà Vu? Rethinking "Contamination-Free'' Dynamic Evaluation for Multimodal Automated Fact-Checking 🧐

[![arXiv](https://img.shields.io/badge/arXiv-2607.23514-B31B1B.svg)](https://arxiv.org/abs/2607.23514)
[![GitHub](https://img.shields.io/badge/GitHub-Rethink_MAFC_Eval-181717.svg)](https://github.com/TrustworthyComp/Rethink-MAFC-Eval)
[![HF Daily Paper](https://img.shields.io/badge/Hugging%20Face-Rethink_MAFC_Eval-yellow.svg)](https://huggingface.co/papers/2607.23514)
[![Website](https://img.shields.io/badge/Project%20Page-Rethink_MAFC_Eval-1DA1F2.svg)](https://trustworthycomp.github.io/Rethink-MAFC-Eval/)


This repository contains the official implementation of the MM-2026 paper, **Novel Claim or Déjà Vu? Rethinking ''Contamination-Free'' Dynamic Evaluation for Multimodal Automated Fact-Checking**.

## News 🔥

- **2026-07-26** 📄 – Preprint released on [arXiv](https://arxiv.org/abs/2607.23514).
- **2026-07-23** 🏆 – Accepted by MM-2026; see you in Brazil! 🇧🇷
- **2026-07-15** 🎉 – Source code is publicly released; contributions and feedback are welcome!

This pipeline extracts evidence from LLM-generated fact-checking articles and measures how similar the extracted evidence is to a reference set, using either **METEOR** or **embedding cosine similarity** (Hungarian matching).

📊 Our curated ClaimReview 2025Q4 dataset is available on Hugging Face: [TrustworthyComp/ClaimReview2025Q4](https://huggingface.co/datasets/TrustworthyComp/ClaimReview2025Q4).

## ⚙️ Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

## 🚀 Quick Start

### 1️⃣ Generate reference evidence

First, run extraction on your dataset to produce reference evidence:

```bash
python extraction/extract.py \
  --mode reference \
  --dataset averitec \
  --extract-model gpt-4o-mini-2024-07-18 \
  --output-dir output/reference \
  --max-items 100
```

### 2️⃣ Run the full pipeline

Then generate evidence with the model and score it against the reference:

```bash
python run.py \
  --generation-model gpt-4o-mini-2024-07-18 \
  --extract-model gpt-4o-mini-2024-07-18 \
  --dataset averitec \
  --similarity-model meteor \
  --ref-json output/reference/averitec_ref_evidence_100_items.json \
  --max-items 100
```

## 🧩 Dataset Options

- `averitec` -> `data/input/averitec.json`
- `claim2025q4` -> `data/input/claim_review_2025q4.json` (default)
- `--input-file <path>` can override `--dataset` in the extraction script

> ⚠️ `run.py` currently supports dataset presets only (`averitec`, `claim2025q4`).
> ⚠️ Generation and reference JSON are matched by exact claim text.

## 📐 Similarity Models

| `--similarity-model` | Method |
|---|---|
| `meteor` | METEOR text similarity + Hungarian matching |
| `Qwen/Qwen3-Embedding-0.6B` | Qwen3 embedding cosine similarity + Hungarian matching |
| `google/embeddinggemma-300m` | GemmaEmbedding cosine similarity + Hungarian matching |

## 🤖 Supported LLM Models (via OpenRouter)

- `gpt-5.2`
- `gpt-4o-mini-2024-07-18`
- `gemini-3.0-pro`
- `gemini-3.0-flash`
- `deepseek-v3.2`
- `qwen3.5-122b-a10b`

## 🔄 Pipeline Steps

1. **Extract reference evidence** (`extraction/extract.py --mode reference`): Extract evidence directly from the oracle fact-checking article (`review_full_article`).
2. **Extract generation evidence** (`extraction/extract.py --mode generation`): Use an LLM to generate a fact-checking article for each claim, then extract evidence sentences from the article.
3. **Score similarity**: Compare generation evidence against reference evidence using the chosen similarity model:
   - 📝 METEOR path: `scoring/meteor.py`
   - 🧮 Embedding path: `scoring/embed.py` (generate embeddings) → `scoring/embed_match.py` (cosine matching)

## 📁 Project Structure

```
extraction/extract.py            Evidence extraction via OpenRouter LLM (--mode generation | reference)
scoring/meteor.py                METEOR-based evidence scoring
scoring/embed.py                 Evidence embedding generation
scoring/embed_match.py           Embedding cosine similarity scoring
run.py                           Unified pipeline runner (generation + scoring)
data/input/                      Input datasets (AVERITeC, ClaimReview 2025Q4)
```

## 🛠️ Individual Script Usage

Each script can be run standalone. Use `--help` for full argument details:

```bash
python extraction/extract.py --help
python scoring/meteor.py --help
python scoring/embed.py --help
python scoring/embed_match.py --help
python run.py --help
```

## 📚 Citation

If you find this work useful, please cite our paper presented at ACM MM 2026:

```bibtex
@inproceedings{rethink_mafc_eval_2026,
  title={Novel Claim or Déjà Vu? Rethinking ''Contamination-Free'' Dynamic Evaluation for Multimodal Automated Fact-Checking},
  author={He, Haorui and Chen, Xinwen and Wen, Dacheng and Cheng, Reynold and Lau, Francis C. M. and Li, Yupeng},
  booktitle={Proc.~of MM},
  year={2026},
}
```

## 🙏 Acknowledgements

- **Data collection pipeline**: We referred to [MisinfoMe](https://github.com/MartinoMensio/MisinfoMe) and [CIMPLE Knowledge Base](https://github.com/CIMPLE-project/knowledge-base) for claim review data collection.
- **Similarity matching**: We referred to [AVeriTeC](https://github.com/MichSchli/AVeriTeC) for evidence similarity evaluation methodology.
- **Datasets**: We use data from the [ClaimReview Project](https://www.claimreviewproject.com/) and [AVeriTeC](https://huggingface.co/chenxwh/AVeriTeC).
