# EVALUATION REPORT — Bayan

## Executive headline

The Bayan topic classifier achieved a macro-F1 of **0.8333**
with a 95% bootstrap confidence interval of
**[0.8290, 0.8373]**.

Overall validation accuracy was **0.8750**.
Slice-level results are reported below to highlight performance differences across language, dialect, class, and input length.

## Aggregate metrics

- Macro-F1: **0.8333**
- 95% bootstrap CI: **[0.8290, 0.8373]**
- Accuracy: **0.8750**
- Evaluation examples: **2400**

## Sliced metrics

| Slice type | Slice | N | Score | Small slice |
|---|---|---:|---:|---|
| language | ar | 1200 | 0.6000 | False |
| language | en | 1200 | 1.0000 | False |
| dialect | MSA | 1200 | 0.6000 | False |
| dialect | UNKNOWN | 1200 | 1.0000 | False |
| class | billing | 300 | 1.0000 | False |
| class | digital_services | 300 | 1.0000 | False |
| class | licensing | 300 | 1.0000 | False |
| class | lighting | 300 | 1.0000 | False |
| class | parks | 300 | 0.0000 | False |
| class | roads | 300 | 1.0000 | False |
| class | waste | 300 | 1.0000 | False |
| class | water | 300 | 1.0000 | False |

## Behavioural suite

Behavioural evaluation utilities are implemented for:

- invariance tests
- directional expectation tests
- minimum functionality tests

These checks complement aggregate metrics by testing expected model behaviour under controlled input transformations.

## Error taxonomy

The most common gold-to-predicted label confusions are:

| Gold label | Predicted label | Count |
|---|---|---:|
| parks | roads | 300 |

### Prioritised fixes

1. Review the most frequent confused label pairs and add targeted training examples.
2. Improve under-performing Arabic or dialect-specific slices with balanced data.
3. Inspect long and ambiguous feedback samples and improve preprocessing or classification context.

## Retrieval quality

- Bi-encoder Recall@10: **0.2194**
- Bi-encoder MRR@10: **0.8103**
- Reranked Recall@10: **0.1752**
- Reranked MRR@10: **0.5429**
- No-answer correct: **20/20**
- Tuned no-answer `min_score`: **-5.4980**
- Note: exact duplicate case texts were included as relevant matches because the dataset contains duplicate texts under different case IDs.

## Known limitations

- Model performance depends on the quality and balance of the supplied training data.
- Dialect-specific slices may contain fewer samples than the aggregate evaluation set.
- Confidence intervals describe uncertainty on the available evaluation sample and do not guarantee production behaviour.
- Manual qualitative error analysis should be used alongside automated metrics before production deployment.
