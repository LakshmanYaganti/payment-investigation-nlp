# Dataset: BANKING77

This project uses **BANKING77** — 13,083 real, anonymised customer-service
queries labelled with **77 banking intents**.

- Paper: Casanueva et al. (2020), *Efficient Intent Detection with Dual
  Sentence Encoders*, NLP4ConvAI workshop — https://arxiv.org/abs/2003.04807
- Hugging Face: [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77)
- Canonical source: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
  (`banking_data/train.csv`, `banking_data/test.csv`)
- License: the dataset is released for research purposes by PolyAI
  (see the dataset card / source repo). **No dataset files are committed to
  this repository.**

## Split

The official split is used unchanged:

| Split | Rows |
|---|---|
| Train | 10,003 |
| Test  | 3,080  |
| Total | 13,083 across 77 intents |

## How the data is loaded

`src/data.py` downloads the data on first use (Hugging Face hub via the
`datasets` library where supported, otherwise the authors' canonical CSVs)
and caches it under `data/cache/` — which is git-ignored. Set the
`BANKING77_CACHE` environment variable to cache it somewhere else.

## Why all 77 intents (no grouping)

Each of the 77 intents maps to a distinct support queue / resolution
playbook (e.g. `card_arrival` vs `lost_or_stolen_card` vs
`card_not_working`). Grouping them into coarse buckets would make the metrics
look better while destroying exactly the routing granularity the system
exists to provide — and would hide the confusable intent pairs that the
error analysis in the notebook focuses on.
