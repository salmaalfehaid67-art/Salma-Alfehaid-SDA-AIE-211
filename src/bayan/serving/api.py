"""Lab 7: Bayan FastAPI service."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from bayan.preprocessing.core import preprocess


MODEL_DIR = Path("artifacts/onnx/classifier_int8")
MODEL_PATH = MODEL_DIR / "model_quantized.onnx"

app = FastAPI(
    title="Bayan — Bilingual Citizen-Feedback Intelligence Service"
)


class TextRequest(BaseModel):
    text: str


tokenizer = None
session = None
config = None


@app.on_event("startup")
def startup_event():
    global tokenizer, session, config

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Classifier ONNX model not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    # Startup canary
    sample = preprocess("مشكلة في إنارة الشارع")

    encoded = tokenizer(
        sample,
        return_tensors="np",
        truncation=True,
        max_length=256,
    )

    input_names = {
        item.name
        for item in session.get_inputs()
    }

    ort_inputs = {
        key: value.astype(np.int64)
        for key, value in encoded.items()
        if key in input_names
    }

    outputs = session.run(None, ort_inputs)

    if outputs is None or len(outputs) == 0:
        raise RuntimeError("Startup canary failed")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier_loaded": session is not None,
        "model": "ONNX INT8",
    }


@app.post("/v1/classify")
def classify(payload: TextRequest):
  if session is None:
    cleaned = preprocess(payload.text)

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Text must not be empty",
        )

    return {
        "text": payload.text,
        "cleaned_text": cleaned,
        "label": "UNKNOWN",
        "confidence": 0.0,
    }

    cleaned = preprocess(payload.text)

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Text must not be empty",
        )

    encoded = tokenizer(
        cleaned,
        return_tensors="np",
        truncation=True,
        max_length=256,
    )

    input_names = {
        item.name
        for item in session.get_inputs()
    }

    ort_inputs = {
        key: value.astype(np.int64)
        for key, value in encoded.items()
        if key in input_names
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    pred_id = int(
        np.argmax(logits, axis=-1)[0]
    )

    label = config.id2label.get(
        pred_id,
        str(pred_id)
    )

    exp_scores = np.exp(
        logits[0] - np.max(logits[0])
    )

    probs = exp_scores / exp_scores.sum()

    confidence = float(
        probs[pred_id]
    )

    return {
        "text": payload.text,
        "cleaned_text": cleaned,
        "label": label,
        "confidence": round(confidence, 4),
    }
