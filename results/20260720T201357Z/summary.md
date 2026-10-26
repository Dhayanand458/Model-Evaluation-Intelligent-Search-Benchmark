# triage-mcp evaluation

Corpus 47804 issues · train 38243 (2025-07-20 → 2026-04-14) · holdout 9561 (2026-04-14 → 2026-07-20)

## Component classification

| method | accuracy | macro-F1 | weighted-F1 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| majority baseline | 0.9485 | **0.1217** | 0.9235 | — | — |
| TF-IDF + logistic regression | 0.7566 | **0.3070** | 0.8300 | 0.92 ms | 2.01 ms |
| embedding kNN (k=10) | 0.9407 | **0.2310** | 0.9274 | 4.97 ms | 6.84 ms |

Best by macro-F1: **TF-IDF + logistic regression**

### Per-class (TF-IDF + logistic regression)

| class | precision | recall | F1 | holdout support |
|---|---:|---:|---:|---:|
| `chat` | 0.020 | 0.232 | 0.036 | 56 |
| `chat-billing` | 0.388 | 0.680 | 0.494 | 122 |
| `chat-terminal` | 0.146 | 0.852 | 0.250 | 54 |
| `ux` | 0.113 | 0.486 | 0.183 | 146 |
| `copilot-cli-agent` | 0.088 | 0.571 | 0.153 | 56 |
| `accessibility` | 0.317 | 0.591 | 0.413 | 44 |
| `git` | 0.036 | 0.500 | 0.066 | 14 |
| `other` | 0.980 | 0.767 | 0.861 | 9069 |

## Retrieval proxy — label-match precision@5

> **PROXY METRIC.** label-match precision@5 is a PROXY for duplicate/similar-issue retrieval. It measures how often a retrieved neighbour shares the query issue's component class, NOT whether it is actually a duplicate. There is no duplicate-pair ground truth in this corpus. Two issues in the same component are usually not duplicates, so treat this as a relative signal for comparing retrieval methods, never as an absolute duplicate-detection rate.

| method | P@5 (all holdout) | P@5 (excl. `other`) | p50 latency | p95 latency |
|---|---:|---:|---:|---:|
| embeddings (all-MiniLM-L6-v2, exact cosine) | 0.8746 | 0.2122 | 4.71 ms | 6.02 ms |
| TF-IDF cosine | 0.8877 | 0.0959 | 160.65 ms | 203.58 ms |

## Task construction

Classes (7 of 10 requested): `chat`, `chat-billing`, `chat-terminal`, `ux`, `copilot-cli-agent`, `accessibility`, `git`, `other`

Demoted to `other` for insufficient holdout support (< 10):

| label | train | holdout |
|---|---:|---:|
| `chat-agents-view` | 416 | 0 |
| `chat-agent` | 272 | 4 |
| `chat-prompts` | 237 | 9 |

`other` composition: 13026 unlabelled, 30696 labelled but with no class label (43722 rows). Accuracy is inflated by this bucket; macro-F1 is the number to compare methods on.

Full mapping and stoplist: `class_map.json`. All numbers: `metrics.json`.
