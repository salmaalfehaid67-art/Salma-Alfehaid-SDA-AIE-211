"""Lab 7: export trained models to ONNX and INT8 without Optimum."""

from pathlib import Path
import shutil

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)


CLASSIFIER_DIR = Path("artifacts/topic_classifier")
NER_DIR = Path("artifacts/ner")
OUTPUT_ROOT = Path("artifacts/onnx")


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


class NERWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def copy_tokenizer_files(tokenizer, target_dir):
    tokenizer.save_pretrained(target_dir)


def export_classifier():
    fp32_dir = OUTPUT_ROOT / "classifier_fp32"
    int8_dir = OUTPUT_ROOT / "classifier_int8"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_DIR
    )
    model.eval()

    wrapper = ClassifierWrapper(model)
    wrapper.eval()
    wrapper.eval()

    sample = tokenizer(
        "اختبار تصنيف بلاغ عربي",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    fp32_model = fp32_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        (
            sample["input_ids"],
            sample["attention_mask"],
        ),
        str(fp32_model),
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "sequence",
            },
            "attention_mask": {
                0: "batch",
                1: "sequence",
            },
            "logits": {
                0: "batch",
            },
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    copy_tokenizer_files(
        tokenizer,
        fp32_dir,
    )

    shutil.copy(
        CLASSIFIER_DIR / "config.json",
        fp32_dir / "config.json",
    )

    int8_model = int8_dir / "model_quantized.onnx"

    quantize_dynamic(
        model_input=str(fp32_model),
        model_output=str(int8_model),
        weight_type=QuantType.QInt8,
    )

    for file in fp32_dir.iterdir():
        if file.name != "model.onnx":
            shutil.copy(
                file,
                int8_dir / file.name,
            )

    print("Classifier FP32:", fp32_model)
    print("Classifier INT8:", int8_model)


def export_ner():
    fp32_dir = OUTPUT_ROOT / "ner_fp32"
    int8_dir = OUTPUT_ROOT / "ner_int8"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        NER_DIR
    )

    model = AutoModelForTokenClassification.from_pretrained(
        NER_DIR
    )
    model.eval()

    wrapper = NERWrapper(model)
    wrapper.eval()
    wrapper.eval()

    sample = tokenizer(
        "تم تسجيل البلاغ في الرياض بتاريخ اليوم",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    fp32_model = fp32_dir / "model.onnx"

    torch.onnx.export(
        wrapper,
        (
            sample["input_ids"],
            sample["attention_mask"],
        ),
        str(fp32_model),
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "sequence",
            },
            "attention_mask": {
                0: "batch",
                1: "sequence",
            },
            "logits": {
                0: "batch",
                1: "sequence",
            },
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    copy_tokenizer_files(
        tokenizer,
        fp32_dir,
    )

    shutil.copy(
        NER_DIR / "config.json",
        fp32_dir / "config.json",
    )

    int8_model = int8_dir / "model_quantized.onnx"

    quantize_dynamic(
        model_input=str(fp32_model),
        model_output=str(int8_model),
        weight_type=QuantType.QInt8,
    )

    for file in fp32_dir.iterdir():
        if file.name != "model.onnx":
            shutil.copy(
                file,
                int8_dir / file.name,
            )

    print("NER FP32:", fp32_model)
    print("NER INT8:", int8_model)


def main():
    if not CLASSIFIER_DIR.exists():
        raise FileNotFoundError(
            "Missing artifacts/topic_classifier"
        )

    if not NER_DIR.exists():
        raise FileNotFoundError(
            "Missing artifacts/ner"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    export_classifier()
    export_ner()

    print("LAB 7 ONNX EXPORT COMPLETE")


if __name__ == "__main__":
    main()
