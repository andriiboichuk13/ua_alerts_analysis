from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

DATASET_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/{}"
)
VOLUNTEER_FILE = "volunteer_data_en.csv"
OFFICIAL_FILE = "official_data_en.csv"

LOCAL_TZ = "Europe/Kyiv"

# Alerts longer than this are treated as a missed "all clear" message and capped.
# Volunteer: above its 99.99th percentile. Official: above the 99.9th percentile of raion-level
# alerts (48 h) - multi-day alerts on the front line are real there.
MAX_ALERT_HOURS = {"volunteer": 12, "official": 72}

# Official alert levels used to define "alert in oblast". Hromada-level alerts are very local
# (single communities near the front) and would make front-line oblasts almost permanently "on".
OFFICIAL_LEVELS = ["oblast", "raion"]

# Modelling regime: raion-level alerting dominates from Nov 2025; the yellow/red two-level
# system (CabMin resolution No. 1092) took effect on 4 Sep 2026 (00:00 Kyiv = 3 Sep 21:00 UTC).
REGIME_START = "2025-11-01 00:00+02:00"
REGIME_END = "2026-09-04 00:00+03:00"

# Forecasting setup: horizons in hours; rolling-origin backtest over the last TEST_DAYS before REGIME_END.
HORIZONS = [1, 3, 6]
TEST_DAYS = 35
N_FOLDS = 4
# Validation folds (Kyiv dates, [start, end)) spread over winter-spring-summer, before the test period.
# Used for hyperparameter tuning and for checking that the model ranking holds across seasons.
VALIDATION_FOLDS = [("2026-02-01", "2026-02-15"), ("2026-03-15", "2026-03-29"),
                    ("2026-04-26", "2026-05-10"), ("2026-06-07", "2026-06-21")]
TUNING_HORIZON = 3
SLIDING_TRAIN_DAYS = 84  # 12 weeks; the sliding-window scheme trains only on this many days before each fold

# Macro-regions for aggregated regional-state features (EDA: alerts propagate within the rear).
MACRO_REGIONS = {
    "west": ["Volynska oblast", "Rivnenska oblast", "Lvivska oblast", "Ternopilska oblast", "Khmelnytska oblast",
             "Ivano-Frankivska oblast", "Zakarpatska oblast", "Chernivetska oblast"],
    "center": ["Vinnytska oblast", "Zhytomyrska oblast", "Kyivska oblast", "Kyiv City", "Cherkaska oblast",
               "Kirovohradska oblast", "Poltavska oblast"],
    "south": ["Odeska oblast", "Mykolaivska oblast", "Khersonska oblast"],
    "front": ["Chernihivska oblast", "Sumska oblast", "Kharkivska oblast", "Donetska oblast",
              "Dnipropetrovska oblast", "Zaporizka oblast"],
}

# Permanent alerts not present in the datasets (see dataset README).
PERMANENT_REGIONS = ["Luhanska oblast", "Crimea"]
