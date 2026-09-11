"""Lab 7: benchmark ONNX classifier inference latency."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


FP32_DIR = Path("artifacts/onnx/classifier_fp32")
INT8_DIR = Path("artifacts/onnx/classifier_int8")
MIX_PATH = Path("data/serving/bench_mix.npy")
OUTPUT_PATH = Path("artifacts/inference_benchmark.json")

WARMUP_RUNS = 20
THREADS = 1


def load_benchmark_texts():
    if MIX_PATH.exists():
        arr = np.load(
            MIX_PATH,
            allow_pickle=True,
        )

        arr = np.asarray(arr).reshape(-1)

        return [
            str(x)
            for x in arr[:200]
        ]

    lengths = [
        16, 32, 64, 96,
        128, 32, 64, 16,
    ] * 20

    return [
        make_text(length)
        for length in lengths
    ]


def make_text(length):
    words = [
        "بلاغ",
        "خدمة",
        "طريق",
        "مياه",
        "إنارة",
        "طلب",
        "مشكلة",
        "الرياض",
    ]

    tokens = [
        words[i % len(words)]
        for i in range(length)
    ]

    return " ".join(tokens)


def create_session(model_path):
    options = ort.SessionOptions()

    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = THREADS

    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark(model_path, tokenizer, texts):
    session = create_session(model_path)

    input_names = {
        item.name
        for item in session.get_inputs()
    }

    warm_text = "بلاغ خدمة طريق مياه إنارة طلب مشكلة الرياض"

    warm = tokenizer(
        warm_text,
        return_tensors="np",
        truncation=True,
        max_length=256,
    )

    warm_inputs = {
        key: value.astype("int64")
        for key, value in warm.items()
        if key in input_names
    }

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            warm_inputs,
        )

    latencies = []

    for text_item in texts:
        encoded = tokenizer(
            str(text_item),
            return_tensors="np",
            truncation=True,
            max_length=256,
        )

        inputs = {
            key: value.astype("int64")
            for key, value in encoded.items()
            if key in input_names
        }

        start_time = time.perf_counter()

        session.run(
            None,
            inputs,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        latencies.append(elapsed_ms)

    return {
        "runs": len(latencies),
        "p50_ms": float(
            np.percentile(latencies, 50)
        ),
        "p99_ms": float(
            np.percentile(latencies, 99)
        ),
        "mean_ms": float(
            np.mean(latencies)
        ),
    }


def main():
    fp32_model = (
        FP32_DIR / "model.onnx"
    )

    int8_model = (
        INT8_DIR / "model_quantized.onnx"
    )

    if not fp32_model.exists():
        raise FileNotFoundError(
            fp32_model
        )

    if not int8_model.exists():
        raise FileNotFoundError(
            int8_model
        )

    tokenizer = AutoTokenizer.from_pretrained(
        FP32_DIR
    )

    texts = load_benchmark_texts()

    print(
        f"Benchmark samples: {len(texts)}"
    )

    print("\nBenchmarking FP32...")
    fp32 = benchmark(
        fp32_model,
        tokenizer,
        texts,
    )

    print("\nBenchmarking INT8...")
    int8 = benchmark(
        int8_model,
        tokenizer,
        texts,
    )

    speedup_p50 = (
        fp32["p50_ms"]
        / int8["p50_ms"]
        if int8["p50_ms"] > 0
        else None
    )

    speedup_p99 = (
        fp32["p99_ms"]
        / int8["p99_ms"]
        if int8["p99_ms"] > 0
        else None
    )

    output = {
        "threads": THREADS,
        "warmup_runs": WARMUP_RUNS,
        "fp32": fp32,
        "int8": int8,
        "speedup_p50": speedup_p50,
        "speedup_p99": speedup_p99,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== FP32 ===")
    print(
        json.dumps(
            fp32,
            indent=2,
        )
    )

    print("\n=== INT8 ===")
    print(
        json.dumps(
            int8,
            indent=2,
        )
    )

    print("\n=== Speedup ===")
    print(
        f"p50 speedup: "
        f"{speedup_p50:.2f}x"
    )
    print(
        f"p99 speedup: "
        f"{speedup_p99:.2f}x"
    )

    print(
        "\nSaved to "
        "artifacts/inference_benchmark.json"
    )


if __name__ == "__main__":
    main()
