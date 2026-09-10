# Bayan — Bilingual Citizen Feedback Intelligence System

This repository contains my implementation of the **SDA AIE-211 course labs** using Google Colab.

Bayan is a bilingual NLP project designed to process and analyse citizen feedback in both Arabic and English. The project combines preprocessing, transformer-based modelling, semantic search, evaluation, optimisation, and API serving in one end-to-end workflow.

## Project Scope

The implementation includes:

- Arabic and English text preprocessing
- PII masking and sentence segmentation
- Tokenizer evaluation
- Scaled Dot-Product Attention
- Multi-Head Attention
- Topic classification
- Named Entity Recognition (NER)
- Extractive Question Answering
- Arabic normalization and dialect analysis
- Semantic search using FAISS
- Cross-encoder reranking
- Evaluation using bootstrap confidence intervals and slice analysis
- ONNX model export
- INT8 quantization
- FastAPI model serving

## Labs Summary

### Lab 1 — Bilingual Preprocessing and Tokenisation
Implemented text normalization, PII masking, sentence segmentation, and tokenizer auditing.

### Lab 2 — Transformer Anatomy
Implemented Scaled Dot-Product Attention, Multi-Head Attention, causal masking, and padding masking.

### Lab 3 — Classification, NER and QA
Implemented a TF-IDF baseline, leakage-safe data splitting, Transformer-based classification, NER, label alignment, and extractive QA.

### Lab 4 — Arabic and Dialect Processing
Implemented Arabic normalization, diacritic and Tatweel handling, token segmentation, and dialect analysis.

### Lab 5 — Semantic Search
Implemented bilingual semantic retrieval using Sentence Transformers, FAISS, and cross-encoder reranking.

### Lab 6 — Evaluation
Implemented bootstrap confidence intervals, slice-based evaluation, and behavioural evaluation utilities.

### Lab 7 — Optimisation and Serving
Implemented ONNX export, INT8 quantization, FastAPI serving, startup checks, and a topic classification endpoint.

## Technologies

Python, Google Colab, PyTorch, Hugging Face Transformers, CAMeLBERT, Sentence Transformers, FAISS, Scikit-learn, Pandas, NumPy, CAMeL Tools, ONNX, ONNX Runtime, FastAPI, and Uvicorn.

## Development Environment

The project was implemented and tested using **Google Colab with Python 3.12**.

## Course Information

This project was completed as part of the **SDA AIE-211 course at SDAIA Academy**.

Original course repository:

https://github.com/AljawharaAlbahlalDev/SDA-AIE-211-Bayan-Course

SDAIA Academy GitHub:

https://github.com/SDAIAAcademy

## Author

**Salma Alfehaid**
