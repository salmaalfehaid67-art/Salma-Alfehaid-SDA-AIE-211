"""Lab 1: bilingual preprocessing."""

import re

PREPROC_VERSION = "1.2.0"


def normalize(text: str) -> str:
    text = text.replace("ـ", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.strip()


def mask_pii(text: str) -> str:
    phone_pattern = r"(?<!\d)(?:\+9665\d{8}|9665\d{8}|05\d{8})(?!\d)"
    text = re.sub(phone_pattern, "<PHONE>", text)

    national_id_pattern = r"(?<!\d)[12]\d{9}(?!\d)"
    text = re.sub(national_id_pattern, "<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    return normalize(mask_pii(text))
