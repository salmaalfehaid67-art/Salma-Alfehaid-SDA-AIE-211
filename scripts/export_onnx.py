"""Lab 7: export trained models to ONNX and INT8."""

from pathlib import Path
import shutil

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,
    ORTModelForTokenClassification,
)
from transformers import AutoTokenizer


CLASSIFIER_DIR = Path("artifacts/topic_classifier")
NER_DIR = Path("artifacts/ner")

OUTPUT_ROOT = Path("artifacts/onnx")


def export_classifier():
    fp32_dir = OUTPUT_ROOT / "classifier_fp32"
    int8_dir = OUTPUT_ROOT / "classifier_int8"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(
        CLASSIFIER_DIR,
        export=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        CLASSIFIER_DIR
    )

    model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)

    fp32_model = fp32_dir / "model.onnx"
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
                int8_dir / file.name
            )

    print("Classifier FP32:", fp32_model)
    print("Classifier INT8:", int8_model)


def export_ner():
    fp32_dir = OUTPUT_ROOT / "ner_fp32"
    int8_dir = OUTPUT_ROOT / "ner_int8"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForTokenClassification.from_pretrained(
        NER_DIR,
        export=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        NER_DIR
    )

    model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)

    fp32_model = fp32_dir / "model.onnx"
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
                int8_dir / file.name
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
        exist_ok=True
    )

    export_classifier()
    export_ner()

    print("LAB 7 ONNX EXPORT COMPLETE")


if __name__ == "__main__":
    main()
