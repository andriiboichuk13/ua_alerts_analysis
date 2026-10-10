"""Scrape a public Telegram channel through its web preview (t.me/s/<channel>), no API credentials needed.

Messages are stored as JSON lines (id, UTC timestamp, text) in data/raw/telegram/<channel>.jsonl; re-running resumes.

Run: uv run python -m ua_alerts.telegram kpszsu --from-id 61000 --to-id 76500
"""
import argparse
import html
import json
import re
import time
import urllib.request

import pandas as pd

from ua_alerts import config

TG_DIR = config.RAW_DIR / "telegram"


def _fetch(channel: str, before: int) -> str:
    req = urllib.request.Request(f"https://t.me/s/{channel}?before={before}", headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {channel} before={before}")


def _parse(page: str) -> list[dict]:
    out = []
    for block in page.split('<div class="tgme_widget_message_wrap')[1:]:
        m_id = re.search(r'data-post="[^/]+/(\d+)"', block)
        m_dt = re.search(r'datetime="([^"]+)"', block)
        m_tx = re.search(r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>', block, re.S)
        if not (m_id and m_dt):
            continue
        text = re.sub(r"<br\s*/?>", "\n", m_tx.group(1)) if m_tx else ""
        out.append({"id": int(m_id.group(1)), "time": m_dt.group(1),
                    "text": html.unescape(re.sub(r"<[^>]+>", "", text)).strip()})
    return out


def scrape(channel: str, from_id: int, to_id: int, delay: float = 0.4) -> pd.DataFrame:
    TG_DIR.mkdir(parents=True, exist_ok=True)
    path = TG_DIR / f"{channel}.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["id"] for line in path.open(encoding="utf-8")}
    before = to_id + 1
    with path.open("a", encoding="utf-8") as f:
        while before > from_id:
            if all(i in seen for i in range(max(from_id, before - 20), before)):
                before -= 20
                continue
            msgs = _parse(_fetch(channel, before))
            if not msgs:
                break
            for m in msgs:
                if m["id"] not in seen and from_id <= m["id"] <= to_id:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    seen.add(m["id"])
            before = min(m["id"] for m in msgs)
            time.sleep(delay)
    return load(channel)


def load(channel: str) -> pd.DataFrame:
    df = pd.read_json(TG_DIR / f"{channel}.jsonl", lines=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("id").sort_values("id").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channel")
    ap.add_argument("--from-id", type=int, required=True)
    ap.add_argument("--to-id", type=int, required=True)
    a = ap.parse_args()
    df = scrape(a.channel, a.from_id, a.to_id)
    print(f"{len(df):,} messages, {df.time.min()} .. {df.time.max()}")
