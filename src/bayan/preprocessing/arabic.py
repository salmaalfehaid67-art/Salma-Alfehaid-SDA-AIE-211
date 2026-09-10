"""Lab 4: Arabic normalization."""

from dataclasses import dataclass
import re

try:
    from camel_tools.tokenizers.word import simple_word_tokenize
except Exception:
    simple_word_tokenize = None


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    if profile.name != "bayan_ar_v1":
        raise ValueError(
            f"Unsupported Arabic profile: {profile.name}"
        )

    text = text.replace("ـ", "")

    if profile.dediacritize:
        text = _ARABIC_DIACRITICS.sub("", text)

    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    return text


def segment(text: str) -> list[str]:
    if simple_word_tokenize is None:
        return text.split()

    return [
        token
        for token in simple_word_tokenize(text)
        if token.strip()
    ]
