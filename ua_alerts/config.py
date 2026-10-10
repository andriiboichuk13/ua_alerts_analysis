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

# Permanent alerts not present in the datasets (see dataset README).
PERMANENT_REGIONS = ["Luhanska oblast", "Crimea"]
