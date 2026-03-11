import re
from typing import Dict, List, Tuple

LANGUAGE_LABELS: Dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tr": "Turkish",
    "uk": "Ukrainian",
}

LANGUAGE_ALIASES: Dict[str, List[str]] = {
    "ar": ["arabic", "العربية"],
    "bg": ["bulgarian", "български"],
    "cs": ["czech", "čeština"],
    "da": ["danish", "dansk"],
    "de": ["german", "deutsch"],
    "el": ["greek", "ελληνικά"],
    "en": ["english"],
    "es": ["spanish", "español", "espanol"],
    "fi": ["finnish", "suomi"],
    "fr": ["french", "français", "francais"],
    "he": ["hebrew", "עברית"],
    "hi": ["hindi", "हिंदी", "हिन्दी"],
    "hr": ["croatian", "hrvatski"],
    "hu": ["hungarian", "magyar"],
    "it": ["italian", "italiano"],
    "nl": ["dutch", "nederlands"],
    "no": ["norwegian", "norsk"],
    "pl": ["polish", "polski"],
    "pt": ["portuguese", "português", "portugues"],
    "ro": ["romanian", "română", "romana"],
    "ru": ["russian", "русский"],
    "sk": ["slovak", "slovenčina", "slovencina"],
    "sl": ["slovenian", "slovenščina", "slovenscina"],
    "sr": ["serbian", "српски"],
    "sv": ["swedish", "svenska"],
    "ta": ["tamil", "தமிழ்"],
    "te": ["telugu", "తెలుగు"],
    "tr": ["turkish", "türkçe", "turkce"],
    "uk": ["ukrainian", "українська"],
}

DETECTION_NORMALIZATION = {
    "ar": "ar",
    "bg": "bg",
    "ca": "es",
    "cs": "cs",
    "cy": "en",
    "da": "da",
    "de": "de",
    "el": "el",
    "en": "en",
    "es": "es",
    "et": "en",
    "fa": "ar",
    "fi": "fi",
    "fr": "fr",
    "gu": "hi",
    "he": "he",
    "hi": "hi",
    "hr": "hr",
    "hu": "hu",
    "id": "en",
    "it": "it",
    "kn": "ta",
    "lt": "en",
    "lv": "en",
    "mk": "bg",
    "ml": "ta",
    "mr": "hi",
    "ne": "hi",
    "nl": "nl",
    "no": "no",
    "pa": "hi",
    "pl": "pl",
    "pt": "pt",
    "ro": "ro",
    "ru": "ru",
    "sk": "sk",
    "sl": "sl",
    "so": "en",
    "sq": "en",
    "sr": "sr",
    "sv": "sv",
    "sw": "en",
    "ta": "ta",
    "te": "te",
    "tl": "en",
    "tr": "tr",
    "uk": "uk",
    "ur": "ar",
    "vi": "en",
}

EXPLICIT_LANGUAGE_PREFIXES = [
    r"(?:answer|reply|respond)\s+in",
    r"in",
    r"reply\s+back\s+in",
    r"respond\s+back\s+in",
]


def get_language_label(code: str) -> str:
    return LANGUAGE_LABELS.get(code, code)


def resolve_target_language(question: str) -> Tuple[str, str]:
    explicit_language = _extract_explicit_language_request(question)
    if explicit_language:
        return explicit_language, f"explicit request for {get_language_label(explicit_language)}"

    detected_language = _detect_language(question)
    return detected_language, f"dominant language detected from the user's question: {get_language_label(detected_language)}"


def _extract_explicit_language_request(question: str) -> str:
    lowered = question.lower()
    for code, aliases in LANGUAGE_ALIASES.items():
        for alias in aliases:
            escaped_alias = re.escape(alias.lower())
            for prefix in EXPLICIT_LANGUAGE_PREFIXES:
                pattern = rf"{prefix}\s+{escaped_alias}\b"
                if re.search(pattern, lowered):
                    return code
    return ""


def _detect_language(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        detected = detect(text)
        normalized = DETECTION_NORMALIZATION.get(detected, "")
        if normalized:
            return normalized
        if detected.startswith("pt"):
            return "pt"
        if detected in LANGUAGE_LABELS:
            return detected
    except Exception:
        pass

    if _contains_script(text, 0x0900, 0x097F):
        return "hi"
    if _contains_script(text, 0x0B80, 0x0BFF):
        return "ta"
    if _contains_script(text, 0x0C00, 0x0C7F):
        return "te"
    if _contains_script(text, 0x0400, 0x04FF):
        return "ru"
    if _contains_script(text, 0x0600, 0x06FF):
        return "ar"
    if _contains_script(text, 0x0370, 0x03FF):
        return "el"

    return "en"


def _contains_script(text: str, start: int, end: int) -> bool:
    return any(start <= ord(char) <= end for char in text)
