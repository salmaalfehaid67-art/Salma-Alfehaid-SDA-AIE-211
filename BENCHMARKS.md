# BENCHMARKS

> Results reported below are from the project runs performed for this repository. Missing values are marked as N/A rather than using course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | N/A | N/A | N/A | N/A | N/A |
| XLM-R | N/A | N/A | N/A | N/A | N/A |
| CAMeLBERT | N/A | N/A | N/A | N/A | N/A |
| DistilBERT | N/A | N/A | N/A | N/A | N/A |

- Golden preprocessing: N/A / 25 passed
- PII masking recall: N/A / 60

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | N/A |
| Topic classifier | macro-F1 | 1.0000 | 0.9958 | N/A |
| NER | entity-F1 | 1.0000 | N/A | ~54 min |
| QA | span/null smoke | 12/12 passed | N/A | ~9 min |

### Notes
- Topic classifier checkpoint: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- Topic classifier frozen-test accuracy: 0.9958
- QA smoke test passed all 12 provided answerable cases.
- The current QA smoke set did not include null/no-answer cases.

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | N/A | N/A | N/A | N/A |
| Arabic dialect-aware | N/A | N/A | N/A | N/A |
| optional third model | N/A | N/A | N/A | N/A |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | 0.2194 | 0.8103 | N/A |
| + cross-encoder rerank | 0.1752 | 0.5429 | N/A |
| cross-lingual slice | See language breakdown below | See language breakdown below | N/A |

### Language breakdown
| Language | Configuration | recall@10 | MRR@10 |
|---|---|---:|---:|
| Arabic | bi-encoder | 0.2196 | 0.8056 |
| Arabic | reranked | 0.1985 | 0.6610 |
| English | bi-encoder | 0.2192 | 0.8143 |
| English | reranked | 0.1552 | 0.4418 |

- no-answer empty-correct: 20 / 20
- tuned no-answer `min_score`: -5.4980
- no-answer classification accuracy: 1.0000
- Observation: cross-encoder reranking reduced Recall@10 and MRR@10 compared with the bi-encoder-only configuration.
- Evaluation note: exact duplicate case texts were included as relevant matches because the dataset contains duplicate texts under different case IDs.

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | 0.8333 [0.8290, 0.8373] | N/A | N/A | N/A |
| dialect-aware | N/A | N/A | N/A | N/A |

- paired comparison verdict: N/A
- error taxonomy top categories: N/A
- top-3 prioritised fixes: N/A

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | N/A | N/A | N/A | N/A |
| fp32 torch @128 dynamic | N/A | N/A | N/A | N/A |
| ONNX fp32 @128 | 83.72 ms | 168.11 ms | Baseline ONNX | N/A |
| ONNX INT8 @128 | 48.97 ms | 202.18 ms | p50 speedup 1.71x; p99 speedup 0.83x | N/A |

### ONNX benchmark details
- Samples: 200
- ONNX FP32 mean latency: 86.38 ms
- ONNX INT8 mean latency: 56.82 ms
- p50 speedup: 1.71x
- p99 speedup: 0.83x

- HTTP p99, 16 concurrent: N/A
- classifier quantisation decision: INT8 improves median and mean latency, but worsens p99 latency; deployment decision should therefore depend on tail-latency requirements.
- NER quantisation decision: ONNX FP32 and INT8 artefacts were successfully generated, but NER latency was not benchmarked separately.
