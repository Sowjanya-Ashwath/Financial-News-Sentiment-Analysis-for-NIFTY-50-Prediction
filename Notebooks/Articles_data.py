"""
Nifty 50 News Collector (GDELT + Guardian Edition)
====================================================
Collects 3-4 years of news articles for all 50 Nifty 50 companies.
Replaces yfinance and GNews with GDELT for deep historical coverage.

Sources:
  1. GDELT        → Global news archive, free, no key needed, 10+ years
  2. The Guardian → Full archive, free key included

Install dependencies:
  pip install pandas requests

Output structure:
  nifty50_news_data/
  ├── RELIANCE/
  │   ├── RELIANCE_news.csv
  │   └── RELIANCE_news.json
  ├── HDFCBANK/
  │   ├── HDFCBANK_news.csv
  │   └── HDFCBANK_news.json
  ├── ... (one folder per company)
  └── all_nifty50_news.csv   ← combined master file
"""

import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GUARDIAN_API_KEY = "f1a72523-34b0-4fb4-bb45-953f7e917118"

# ── Change YEARS_BACK to however far back you want (3 or 4 recommended) ──────
YEARS_BACK = 4

END_DATE   = datetime.today() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=365 * YEARS_BACK)

# All 50 Nifty 50 companies
COMPANIES = {
    "TCS.NS":        "Tata Consultancy Services",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS":  "Tata Steel",
    "TECHM.NS":      "Tech Mahindra",
}

OUTPUT_DIR = Path("nifty50_news_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── GDELT settings ────────────────────────────────────────────────────────────
# GDELT allows max 250 records per request.
# We split the date range into quarterly chunks to maximize coverage.
GDELT_BASE_URL   = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250

# ── Guardian settings ─────────────────────────────────────────────────────────
GUARDIAN_BASE_URL  = "https://content.guardianapis.com/search"
MAX_GUARDIAN_PAGES = 20   # increased from 5 to capture more articles
GUARDIAN_PAGE_SIZE = 10


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT NAME ALIASES  (for better search hits)
# ═══════════════════════════════════════════════════════════════════════════════

SHORT_NAMES = {
    "Adani Enterprises":               "Adani Enterprises",
    "Adani Ports":                     "Adani Ports",
    "Apollo Hospitals":                "Apollo Hospitals",
    "Bajaj Finance":                   "Bajaj Finance",
    "Bajaj Finserv":                   "Bajaj Finserv",
    "Bharat Petroleum":                "BPCL",
    "Bharti Airtel":                   "Airtel",
    "Britannia Industries":            "Britannia",
    "Divi's Laboratories":             "Divi's Labs",
    "Dr. Reddy's Laboratories":        "Dr Reddy's",
    "Grasim Industries":               "Grasim",
    "HCL Technologies":                "HCL Tech",
    "HDFC Bank":                       "HDFC Bank",
    "HDFC Life Insurance":             "HDFC Life",
    "Hero MotoCorp":                   "Hero MotoCorp",
    "Hindalco Industries":             "Hindalco",
    "Hindustan Unilever":              "HUL",
    "Indian Oil Corporation":          "IOC",
    "JSW Steel":                       "JSW Steel",
    "Kotak Mahindra Bank":             "Kotak Bank",
    "Larsen & Toubro":                 "L&T",
    "Mahindra & Mahindra":             "M&M",
    "Maruti Suzuki":                   "Maruti",
    "Nestle India":                    "Nestle India",
    "Oil and Natural Gas Corporation": "ONGC",
    "Power Grid Corporation of India": "Power Grid",
    "Reliance Industries":             "Reliance",
    "SBI Life Insurance":              "SBI Life",
    "State Bank of India":             "SBI",
    "Sun Pharmaceutical":              "Sun Pharma",
    "Tata Consultancy Services":       "TCS",
    "Tata Consumer Products":          "Tata Consumer",
    "UltraTech Cement":                "UltraTech",
    "LTIMindtree":                     "LTI Mindtree",
    "SBI Cards and Payment Services":  "SBI Card",
}

def get_short_name(company_name: str) -> str:
    return SHORT_NAMES.get(company_name, company_name)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())

def safe_get(d, *keys, default=""):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d if d else default

def get_symbol(ticker: str) -> str:
    return ticker.replace(".NS", "").replace("&", "AND").replace("-", "_")

def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def quarterly_chunks(start: datetime, end: datetime):
    """
    Split a date range into ~90-day chunks.
    GDELT has a 250-record cap per request; smaller windows = more articles.
    """
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=90), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + timedelta(seconds=1)
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — GDELT
# ═══════════════════════════════════════════════════════════════════════════════
#
# GDELT v2 Doc API docs:
#   https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
#
# Key parameters:
#   query          → search string (supports AND / OR / phrase quotes)
#   mode           → "artlist" returns a list of article URLs + metadata
#   maxrecords     → 1–250 (hard cap)
#   startdatetime  → YYYYMMDDHHMMSS
#   enddatetime    → YYYYMMDDHHMMSS
#   format         → "json"
#   sort           → "DateDesc"
#   sourcelang     → "english" (filter non-English results)
#
# Strategy:
#   • Run multiple focused queries per company
#   • Split the full date range into quarterly chunks
#   • Deduplicate by URL across all queries + chunks

def fetch_gdelt_articles(company_name: str, ticker: str) -> list[dict]:
    sname      = get_short_name(company_name)
    nse_symbol = ticker.replace(".NS", "")

    # Multiple focused queries improve recall
    queries = [
        f'"{company_name}" India stock',
        f'"{sname}" NSE shares',
        f'"{sname}" India results earnings',
        f'"{nse_symbol}" India stock market',
    ]

    # Remove duplicate queries (happens when short name == full name)
    queries = list(dict.fromkeys(queries))

    chunks = quarterly_chunks(START_DATE, END_DATE)
    seen, articles = set(), []

    for query in queries:
        for chunk_start, chunk_end in chunks:
            try:
                params = {
                    "query":         query,
                    "mode":          "artlist",
                    "maxrecords":    GDELT_MAX_RECORDS,
                    "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
                    "enddatetime":   chunk_end.strftime("%Y%m%d%H%M%S"),
                    "format":        "json",
                    "sort":          "DateDesc",
                    "sourcelang":    "english",
                }

                resp = requests.get(GDELT_BASE_URL, params=params, timeout=20)
                resp.raise_for_status()

                data = resp.json()
                for item in data.get("articles", []):
                    url = item.get("url", "")
                    if not url or url in seen:
                        continue
                    seen.add(url)

                    # GDELT seendate format: YYYYMMDDTHHMMSSZ
                    raw_date = item.get("seendate", "")
                    try:
                        pub_dt = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ")
                        pub_str = pub_dt.strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pub_str = raw_date

                    articles.append({
                        "source":       "gdelt",
                        "company":      company_name,
                        "ticker":       ticker,
                        "title":        clean_text(item.get("title", "")),
                        "description":  "",           # GDELT doesn't provide summaries
                        "url":          url,
                        "published_at": pub_str,
                        "publisher":    item.get("domain", ""),
                        "language":     item.get("language", ""),
                    })

                # Be polite to GDELT — it's a free public API
                time.sleep(0.5)

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("    [gdelt]     Rate limited — waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"    [gdelt]     HTTP ERROR — {e}")
            except requests.exceptions.Timeout:
                print(f"    [gdelt]     Timeout on '{query}' chunk {chunk_start.date()} — skipping")
            except Exception as e:
                print(f"    [gdelt]     ERROR — {e}")

        time.sleep(1)  # pause between queries

    print(f"    [gdelt]     {len(articles):>4} articles")
    return articles


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — The Guardian
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_guardian_articles(company_name: str, ticker: str) -> list[dict]:
    sname   = get_short_name(company_name)
    queries = [
        f'"{company_name}"',
        f'"{sname}" India stock',
        f'"{sname}" shares NSE',
    ]

    seen, articles = set(), []

    for query in queries:
        for page in range(1, MAX_GUARDIAN_PAGES + 1):
            try:
                params = {
                    "q":           query,
                    "from-date":   START_DATE.strftime("%Y-%m-%d"),
                    "to-date":     END_DATE.strftime("%Y-%m-%d"),
                    "page":        page,
                    "page-size":   GUARDIAN_PAGE_SIZE,
                    "show-fields": "trailText,byline",
                    "order-by":    "newest",
                    "api-key":     GUARDIAN_API_KEY,
                }
                resp = requests.get(GUARDIAN_BASE_URL, params=params, timeout=10)
                resp.raise_for_status()

                data    = resp.json().get("response", {})
                results = data.get("results", [])

                if not results:
                    break

                for item in results:
                    url = item.get("webUrl", "")
                    if url in seen:
                        continue
                    seen.add(url)
                    fields = item.get("fields", {})
                    articles.append({
                        "source":       "guardian",
                        "company":      company_name,
                        "ticker":       ticker,
                        "title":        clean_text(item.get("webTitle", "")),
                        "description":  clean_text(fields.get("trailText", "")),
                        "url":          url,
                        "published_at": item.get("webPublicationDate", ""),
                        "publisher":    "The Guardian",
                        "section":      item.get("sectionName", ""),
                        "author":       fields.get("byline", ""),
                    })

                time.sleep(0.5)

                if len(results) < GUARDIAN_PAGE_SIZE:
                    break   # no more pages

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print("    [guardian]  Rate limited — waiting 30s...")
                    time.sleep(30)
                else:
                    print(f"    [guardian]  HTTP ERROR — {e}")
                break
            except Exception as e:
                print(f"    [guardian]  ERROR — {e}")
                break

        time.sleep(1)

    print(f"    [guardian]  {len(articles):>4} articles")
    return articles


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE PER-COMPANY FILES
# ═══════════════════════════════════════════════════════════════════════════════

def save_company_files(ticker: str, company_name: str, articles: list[dict]):
    symbol = get_symbol(ticker)
    folder = OUTPUT_DIR / symbol
    folder.mkdir(exist_ok=True)

    if not articles:
        print(f"  ⚠  No articles found for {company_name}")
        return

    df = pd.DataFrame(articles)
    df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"]         = df["published_at"].dt.date
    df.sort_values("published_at", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    csv_path  = folder / f"{symbol}_news.csv"
    json_path = folder / f"{symbol}_news.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_json = df.copy()
    df_json["published_at"] = df_json["published_at"].astype(str)
    df_json["date"]         = df_json["date"].astype(str)
    df_json.to_json(json_path, orient="records", indent=2, force_ascii=False)

    valid_dates = df["date"].dropna()
    date_range  = f"{valid_dates.min()} → {valid_dates.max()}" if not valid_dates.empty else "N/A"

    print(f"\n  ✓  {len(df)} unique articles  |  {date_range}")
    print(f"     CSV  → {csv_path}")
    print(f"     JSON → {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def collect_all_news() -> pd.DataFrame:
    all_articles = []

    print(f"\n{'═'*60}")
    print(f"  Nifty 50 News Collector  (GDELT + Guardian Edition)")
    print(f"  Period    :  {START_DATE.date()}  →  {END_DATE.date()}  ({YEARS_BACK} years)")
    print(f"  Companies :  {len(COMPANIES)}")
    print(f"  Sources   :  GDELT  |  The Guardian")
    print(f"{'═'*60}")

    for i, (ticker, name) in enumerate(COMPANIES.items(), 1):
        symbol = get_symbol(ticker)
        folder = OUTPUT_DIR / symbol

        if folder.exists():
            csv_files = list(folder.glob("*_news.csv"))
            
            if csv_files:
                try:
                    df_check = pd.read_csv(csv_files[0])
                    if len(df_check) > 10:
                        print(f"Skipping {name} (already done)")
                        continue
                    else:
                        print(f"Reprocessing {name} (incomplete data)")
                except:
                    print(f"Reprocessing {name} (corrupted file)")

        print_section(f"[{i}/{len(COMPANIES)}]  {name}  ({ticker})")

        gdelt_arts    = fetch_gdelt_articles(name, ticker)
        guardian_arts = fetch_guardian_articles(name, ticker)

        combined = gdelt_arts + guardian_arts
        all_articles.extend(combined)

        save_company_files(ticker, name, combined)
        time.sleep(2)
    # ── Master DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(all_articles)
    if df.empty:
        print("\n⚠  No articles collected. Check your internet connection.")
        return df

    df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["date"]         = df["published_at"].dt.date
    df.sort_values("published_at", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def save_master(df: pd.DataFrame):
    csv_path  = OUTPUT_DIR / "all_nifty50_news.csv"
    json_path = OUTPUT_DIR / "all_nifty50_news.json"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    df_json = df.copy()
    df_json["published_at"] = df_json["published_at"].astype(str)
    df_json["date"]         = df_json["date"].astype(str)
    df_json.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"\n{'═'*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  Total unique articles : {len(df)}")
    print(f"  Master CSV  →  {csv_path}")
    print(f"  Master JSON →  {json_path}")
    print(f"{'═'*60}")

    summary = df.groupby("company").agg(
        total    = ("title", "count"),
        gdelt    = ("source", lambda x: (x == "gdelt").sum()),
        guardian = ("source", lambda x: (x == "guardian").sum()),
    ).sort_values("total", ascending=False)

    print("\n  Articles per company:\n")
    print(summary.to_string())

    valid_dates = df["date"].dropna()
    if not valid_dates.empty:
        print(f"\n  Earliest article : {valid_dates.min()}")
        print(f"  Latest article   : {valid_dates.max()}\n")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = collect_all_news()
    if not df.empty:
        save_master(df)