from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests


FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]

# Ken French's official library uses "Developed_5_Factors" for the broad
# developed-markets dataset. The existing UI label remains "Global".
DATASET_CONFIG = {
    "US": {
        "reader_name": "F-F_Research_Data_5_Factors_2x3",
        "zip_name": "F-F_Research_Data_5_Factors_2x3",
    },
    "Japan": {
        "reader_name": "Japan_5_Factors",
        "zip_name": "Japan_5_Factors",
    },
    "Global": {
        "reader_name": "Developed_5_Factors",
        "zip_name": "Developed_5_Factors",
    },
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "factors"


def _month_end_index(index: pd.Index) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index))
    if dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    return dt_index.to_period("M").to_timestamp(how="end").normalize()


def _normalize_factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [str(col).strip() for col in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.loc[:, [col for col in FACTOR_COLUMNS if col in frame.columns]]
    if frame.empty:
        raise ValueError("No factor columns were found in the dataset.")

    max_abs = frame.abs().max().max()
    if pd.notna(max_abs) and max_abs > 1.0:
        frame = frame / 100.0

    if "RF" not in frame.columns:
        frame["RF"] = 0.0

    frame.index = _month_end_index(frame.index)
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    return frame.interpolate(method="linear").ffill().bfill()


def _parse_factor_text(text: str) -> pd.DataFrame:
    lines = [line.rstrip() for line in text.splitlines()]
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if "Mkt-RF" in line and "SMB" in line and "HML" in line
        ),
        None,
    )
    if header_idx is None:
        raise ValueError("Could not find the five-factor header row.")

    headers = [col.strip() for col in lines[header_idx].split(",")]
    rows: list[list[str]] = []

    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if rows:
                break
            continue

        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0].isdigit():
            if rows:
                break
            continue

        rows.append(parts[: len(headers)])

    if not rows:
        raise ValueError("No factor rows were found in the downloaded file.")

    df = pd.DataFrame(rows, columns=headers)
    date_col = headers[0]
    sample = str(df[date_col].iloc[0]).strip()

    if len(sample) == 6:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m")
    elif len(sample) == 8:
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d")
    else:
        raise ValueError(f"Unsupported factor date format: {sample}")

    df.set_index(date_col, inplace=True)
    return _normalize_factor_frame(df)


def _cache_path(region: str) -> Path:
    return CACHE_DIR / f"{region.lower()}_5_factors.csv"


def _load_from_cache(region: str) -> pd.DataFrame:
    cache_path = _cache_path(region)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file was not found for region '{region}'.")

    df = pd.read_csv(cache_path, index_col=0, parse_dates=[0])
    return _normalize_factor_frame(df)


def _save_to_cache(region: str, df: pd.DataFrame) -> None:
    cache_path = _cache_path(region)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)


def _load_from_datareader(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    import pandas_datareader.data as web

    config = DATASET_CONFIG[region]
    ff_dict = web.DataReader(
        config["reader_name"],
        "famafrench",
        start=start_date,
        end=end_date,
    )
    if not ff_dict or 0 not in ff_dict:
        raise ValueError(f"Empty response from pandas-datareader for region '{region}'.")

    return _normalize_factor_frame(ff_dict[0])


def _load_from_official_zip(region: str) -> pd.DataFrame:
    config = DATASET_CONFIG[region]
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{config['zip_name']}_CSV.zip"
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        csv_name = next(
            (name for name in zf.namelist() if name.lower().endswith(".csv")),
            None,
        )
        if csv_name is None:
            raise ValueError("Could not find CSV inside the official ZIP archive.")
        text = zf.read(csv_name).decode("utf-8", errors="ignore")

    return _parse_factor_text(text)


def load_factor_dataset(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    normalized_region = region if region in DATASET_CONFIG else "US"

    try:
        df = _load_from_datareader(normalized_region, start_date, end_date)
        df = df.loc[start_date:end_date]
        _save_to_cache(normalized_region, df)
        df.attrs["factor_data_source"] = "datareader"
        return df
    except Exception:
        pass

    try:
        df = _load_from_official_zip(normalized_region)
        df = df.loc[start_date:end_date]
        _save_to_cache(normalized_region, df)
        df.attrs["factor_data_source"] = "official_zip"
        return df
    except Exception:
        pass

    df = _load_from_cache(normalized_region)
    df = df.loc[start_date:end_date]
    df.attrs["factor_data_source"] = "local_cache"
    return df
