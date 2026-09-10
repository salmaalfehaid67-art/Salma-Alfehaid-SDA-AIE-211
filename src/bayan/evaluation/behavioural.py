"""Lab 6: behavioural evaluation."""


def invariance_test(predict_fn, original_texts, transformed_texts):
    if len(original_texts) != len(transformed_texts):
        raise ValueError("Inputs must have the same length")

    passed = 0

    for original, transformed in zip(original_texts, transformed_texts):
        if predict_fn(original) == predict_fn(transformed):
            passed += 1

    total = len(original_texts)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }


def directional_test(
    score_fn,
    original_texts,
    transformed_texts,
    direction="increase",
):
    if len(original_texts) != len(transformed_texts):
        raise ValueError("Inputs must have the same length")

    passed = 0

    for original, transformed in zip(original_texts, transformed_texts):
        a = score_fn(original)
        b = score_fn(transformed)

        if direction == "increase":
            ok = b > a
        elif direction == "decrease":
            ok = b < a
        else:
            raise ValueError("direction must be increase or decrease")

        if ok:
            passed += 1

    total = len(original_texts)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }


def minimum_functionality_test(predict_fn, examples):
    passed = 0
    details = []

    for text, expected in examples:
        prediction = predict_fn(text)
        ok = prediction == expected

        if ok:
            passed += 1

        details.append({
            "text": text,
            "expected": expected,
            "prediction": prediction,
            "passed": ok,
        })

    total = len(examples)

    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "details": details,
    }
