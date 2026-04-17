#!/usr/bin/env python3

from lingua import Language, LanguageDetectorBuilder

NEMO_TO_LINGUA: dict[str, Language] = {
    "bg": Language.BULGARIAN,
    "hr": Language.CROATIAN,
    "cs": Language.CZECH,
    "da": Language.DANISH,
    "nl": Language.DUTCH,
    "en": Language.ENGLISH,
    "et": Language.ESTONIAN,
    "fi": Language.FINNISH,
    "fr": Language.FRENCH,
    "de": Language.GERMAN,
    "el": Language.GREEK,
    "hu": Language.HUNGARIAN,
    "it": Language.ITALIAN,
    "lv": Language.LATVIAN,
    "lt": Language.LITHUANIAN,
    "pl": Language.POLISH,
    "pt": Language.PORTUGUESE,
    "ro": Language.ROMANIAN,
    "sk": Language.SLOVAK,
    "sl": Language.SLOVENE,
    "es": Language.SPANISH,
    "sv": Language.SWEDISH,
    "ru": Language.RUSSIAN,
    "uk": Language.UKRAINIAN,
}

LINGUA_TO_NEMO: dict[Language, str] = {v: k for k, v in NEMO_TO_LINGUA.items()}

def build_detector(lang_codes: list[str]):
    languages = [NEMO_TO_LINGUA[c] for c in lang_codes if c in NEMO_TO_LINGUA]

    return LanguageDetectorBuilder.from_languages(*languages).build()