"""Hourly features from Air Force (@kpszsu) Telegram messages.

A message belongs to hour t if it was published in [t, t+1h), i.e. before the forecast time T = t + 1h.
Hours before the first scraped message are NaN (no data), later hours without messages are 0.
"""
import re

import pandas as pd

# Ukrainian stems of oblast names and their main cities, as used in Air Force messages
# ("Вінниччина: реактивний БпЛА у напрямку Козятина", "... з Хмельниччини в бік Збаража").
OBLAST_PATTERNS = {
    "Vinnytska oblast": r"Вінниччин|Вінниц", "Volynska oblast": r"Волин", "Dnipropetrovska oblast": r"Дніпропетровщин|Дніпропетровськ",
    "Donetska oblast": r"Донеччин|Донецьк", "Zhytomyrska oblast": r"Житомирщин|Житомир", "Zakarpatska oblast": r"Закарпатт",
    "Zaporizka oblast": r"Запоріжж|Запорізьк", "Ivano-Frankivska oblast": r"Прикарпатт|Франківщин|Івано-Франківськ",
    "Kyivska oblast": r"Київщин", "Kirovohradska oblast": r"Кіровоградщин|Кропивницьк", "Lvivska oblast": r"Львівщин|Львів",
    "Mykolaivska oblast": r"Миколаївщин|Миколаїв", "Odeska oblast": r"Одещин|Одес", "Poltavska oblast": r"Полтавщин|Полтав",
    "Rivnenska oblast": r"Рівненщин|Рівн[еоому]", "Sumska oblast": r"Сумщин|\bСум[иах]", "Ternopilska oblast": r"Тернопільщин|Тернопіл",
    "Kharkivska oblast": r"Харківщин|Харків", "Khersonska oblast": r"Херсонщин|Херсон", "Khmelnytska oblast": r"Хмельниччин|Хмельницьк",
    "Cherkaska oblast": r"Черкащин|Черкас", "Chernivetska oblast": r"Буковин|Чернівеччин|Чернівц", "Chernihivska oblast": r"Чернігівщин|Чернігів",
    "Kyiv City": r"Ки[їє]в(?!щин)|столиц",
}
CATEGORIES = {
    "drones": r"БпЛА|шахед|дрон",
    "ballistic": r"балістичн",
    "missiles": r"крилат|\bКР\b|Калібр|Х-?101|Х-?22|Кинджал|МіГ-?31|Ту-?95|Ту-?160|Ту-?22|стратегічн",
    "kab": r"КАБ",
}


def _hourly_counts(times: pd.Series, index: pd.DatetimeIndex, first_hour: pd.Timestamp) -> pd.Series:
    counts = times.dt.floor("h").value_counts().reindex(index, fill_value=0).astype(float)
    counts[index < first_hour] = float("nan")
    return counts


def hourly_text_state(messages: pd.DataFrame, index: pd.DatetimeIndex, regions) -> dict[str, pd.DataFrame | pd.Series]:
    """mentions: hour x region message counts; categories: hour x category national message counts."""
    first_hour = messages["time"].min().ceil("h")       # the first (possibly partial) hour is treated as missing
    mentions = pd.DataFrame({
        r: _hourly_counts(messages.loc[messages["text"].str.contains(OBLAST_PATTERNS[r], regex=True), "time"],
                          index, first_hour) if r in OBLAST_PATTERNS else pd.Series(float("nan"), index=index)
        for r in regions
    })
    categories = pd.DataFrame({
        c: _hourly_counts(messages.loc[messages["text"].str.contains(p, regex=True), "time"], index, first_hour)
        for c, p in CATEGORIES.items()
    })
    return {"mentions": mentions, "categories": categories}
