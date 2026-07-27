"""
Loads admissions data from a Google Sheet.

Default path (used here): reads a PUBLIC sheet (Anyone with the link -> Viewer)
via Google's CSV export endpoint. No API key or credentials needed.

Expected sheet columns (case-insensitive, any order):
    Year        e.g. 2023
    Season      e.g. Spring / Summer / Autumn / Fall / Winter (any 2 of these,
                consistently used -- this app models a 2-season-per-year cycle)
    Admissions  e.g. 124

IMPORTANT: this app supports exactly 2 distinct season names in your sheet
(whichever two you actually use -- Spring/Autumn, Summer/Winter, etc.). The
season dummy variable and the future-period generator are both derived from
whatever season names are actually present in your data, not hardcoded to
"Summer" -- so this works correctly regardless of which two seasons you use.

If your sheet is private, see `load_sheet_data_private()` at the bottom,
which uses a service account via gspread instead.
"""

from typing import Tuple, List
import io
import requests
import pandas as pd

# Natural calendar ordering, used only to decide which of your 2 seasons
# comes first within a year (e.g. Spring before Autumn). Add names here if
# your sheet uses a term not listed.
SEASON_CALENDAR_ORDER = {
    "spring": 0,
    "summer": 1,
    "autumn": 2,
    "fall": 2,
    "winter": 3,
}


def _process(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"year", "season", "admissions"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Sheet must contain columns {required}. Found: {list(df.columns)}"
        )

    df["season"] = df["season"].astype(str).str.strip().str.lower()
    df["year"] = df["year"].astype(int)
    df["admissions"] = df["admissions"].astype(float)

    unique_seasons = sorted(
        df["season"].unique(),
        key=lambda s: SEASON_CALENDAR_ORDER.get(s, 99),
    )
    if len(unique_seasons) != 2:
        raise ValueError(
            "This app models a 2-season-per-year cycle, but found "
            f"{len(unique_seasons)} distinct season name(s) in your sheet: "
            f"{unique_seasons}. Make sure the Season column uses exactly two "
            "consistent values (e.g. always 'Spring'/'Autumn', not a mix of "
            "spellings)."
        )
    season_cycle = unique_seasons  # e.g. ["spring", "autumn"], in calendar order

    df["season_order"] = df["season"].map(lambda s: season_cycle.index(s))
    # season_idx: 0 for the earlier-in-year season, 1 for the later one.
    # Replaces the old hardcoded "is_summer" -- works for any 2 season names.
    df["season_idx"] = df["season_order"]

    df = df.sort_values(["year", "season_order"]).reset_index(drop=True)
    df["t"] = range(len(df))
    df["period_label"] = df["year"].astype(str) + " " + df["season"].str.capitalize()

    return df[
        ["t", "year", "season", "season_idx", "period_label", "admissions"]
    ], season_cycle


def load_sheet_data(
    sheet_id: str, sheet_name: str = "Sheet1"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch and parse data from a public Google Sheet.

    sheet_id: the long ID from the sheet URL
        https://docs.google.com/spreadsheets/d/<THIS_PART>/edit
    sheet_name: the tab name (not the gid number)

    Returns (df, season_cycle) where season_cycle is the 2 season names
    found in the data, in calendar order, e.g. ["spring", "autumn"].
    """
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise ValueError(
            f"Could not fetch sheet (status {resp.status_code}). "
            "Check the sheet ID, tab name, and that sharing is set to "
            "'Anyone with the link can view'."
        )

    df = pd.read_csv(io.StringIO(resp.text))
    return _process(df)


def load_sheet_data_private(
    sheet_id: str, sheet_name: str, credentials_path: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Alternative loader for a PRIVATE sheet using a Google service account.

    Setup:
      1. pip install gspread google-auth
      2. Create a service account in Google Cloud Console, download its JSON key.
      3. Share the Google Sheet with the service account's email (as Viewer).
      4. Pass the path to the downloaded JSON key as `credentials_path`.

    Returns (df, season_cycle), same as load_sheet_data().
    """
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(sheet_name)
    records = ws.get_all_records()
    df = pd.DataFrame(records)

    return _process(df)
