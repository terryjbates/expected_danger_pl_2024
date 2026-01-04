"""Streamlit app to browse position-ranked xD tables (Great Tables).

What this app expects in the repo (default):

    outputs/rankings_partitioned/
        position=Winger/part-0.parquet
        position=Forward/part-0.parquet
        ...

That layout is a standard "Hive partition" directory structure created when
writing a parquet dataset partitioned by the `position` column.

You can change the base directory via the STREAMLIT_RANKINGS_DIR env var.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover
    ds = None  # type: ignore

from great_tables import GT, loc, style


# -----------------------------
# App config
# -----------------------------

st.set_page_config(
    page_title="Premier League 2024 xD Rankings",
    page_icon="⚠️",
    layout="wide",
)

st.markdown(
    """
<style>
/* Wrap GT HTML in <div class="gt-dark"> ... </div> */
.gt-dark {
  background: #0e1117;          /* Streamlit dark-ish */
  color: #e6e6e6;
  padding: 0.75rem 0.75rem;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.08);
  overflow-x: auto;
}

/* GT tables often use <table> + <th>/<td> */
.gt-dark table {
  background: transparent !important;
  color: #e6e6e6 !important;
}

/* header cells */
.gt-dark thead th {
  background: rgba(255,255,255,0.06) !important;
  color: #ffffff !important;
  border-color: rgba(255,255,255,0.10) !important;
}

/* body cells */
.gt-dark tbody td {
  background: transparent !important;
  color: #e6e6e6 !important;
  border-color: rgba(255,255,255,0.10) !important;
}

/* row hover (optional) */
.gt-dark tbody tr:hover td {
  background: rgba(255,255,255,0.04) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

def _rankings_root() -> Path:
    """Where the partitioned parquet rankings live."""
    return Path(os.getenv("STREAMLIT_RANKINGS_DIR", "outputs/rankings_partitioned"))


# -----------------------------
# Data loading
# -----------------------------


def _partition_positions_hive(root: Path) -> list[str]:
    """Return positions from a hive-partitioned directory like `position=Foo/`."""
    if not root.exists():
        return []
    out: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        # expected: position=Winger
        if "=" in child.name:
            key, val = child.name.split("=", 1)
            if key == "position" and val:
                out.append(val)
    return sorted(set(out))


@st.cache_data(show_spinner=False)
def load_rankings_df(root: str) -> pd.DataFrame:
    """Load the entire ranking dataset (all positions) as a DataFrame.

    This is usually small enough to load in memory (a few thousand rows).
    If you prefer lazy reads, you can switch to per-position read below.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Rankings directory not found: {root_path.resolve()}. "
            "Expected something like outputs/rankings_partitioned/"
        )

    # Prefer Arrow dataset reading when available.
    if ds is not None:
        dataset = ds.dataset(str(root_path), format="parquet", partitioning="hive")
        table = dataset.to_table()
        return table.to_pandas()

    # Fallback: glob all parquet parts.
    parts = list(root_path.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found under: {root_path.resolve()}")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def _ensure_required_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal safety: ensure the columns we rely on exist."""
    required = {
        "position",
        "rank_xD",
        "short_name",
        "team",
        "total_min_played",
        "xD_per_90",
        "danger_passes_per_90",
        "total_xD",
        "danger_passes",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Ranking parquet is missing required columns: "
            + ", ".join(missing)
            + "\n\n"
            "Fix: regenerate rankings with these columns, or update the table builder."
        )
    return df


# -----------------------------
# Image handling (cached)
# -----------------------------


@st.cache_data(show_spinner=False)
def fetch_image_as_data_url(url: str, timeout: int = 10) -> str | None:
    """Download an image once, return a data: URL for embedding.

    This avoids re-hitting the CDN on every app rerun.

    Returns None if url is falsy or fetch fails.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "image/png")
        b64 = base64.b64encode(r.content).decode("ascii")
        return f"data:{content_type};base64,{b64}"
    except Exception:
        return None


def md_img(data_url: str | None, size_px: int) -> str:
    """Return markdown for an inline image (or empty string)."""
    if not data_url:
        return ""
    # Use HTML to control size reliably inside GT.
    return (
        f"<img src=\"{data_url}\" width=\"{size_px}\" "
        f"style=\"vertical-align:middle;\"/>"
    )


# -----------------------------
# Table builder
# -----------------------------


def make_position_table(
    df: pd.DataFrame,
    *,
    position: str,
    page: int,
    page_size: int,
    min_minutes: int = 450,
) -> GT:
    """Build a Great Tables ranking table for a single position, paginated.

    Color rules:
      - xD_per_90: Reds (top = dark/bright)
      - danger_passes_per_90: Greens (top = dark/bright)
      - xD_efficiency: YlOrBr (top = dark/bright)
      - no other columns colored

    Images:
      - optional player/team image columns if URLs are present
      - images are cached as data URLs using Streamlit cache
    """
    d = df.copy()

    # Filters
    d = d.loc[d["position"].eq(position)]
    d = d.loc[d["total_min_played"].fillna(0) >= min_minutes]

    # Derived metric (Concept 2)
    d["xD_efficiency"] = d["total_xD"] / d["danger_passes"].replace({0: pd.NA})

    # Rank order (rank_xD is already partition-safe in query)
    d = d.sort_values(["rank_xD"], ascending=True)

    # Pagination (page is 1-based in UI)
    page = max(int(page), 1)
    page_size = max(int(page_size), 1)
    start = (page - 1) * page_size
    end = start + page_size
    d = d.iloc[start:end]

    # Images: build HTML strings; keep it AFTER pagination to minimize downloads
    if "player_url" in d.columns:
        d["player_img"] = d["player_url"].apply(lambda u: md_img(fetch_image_as_data_url(u), 28))
    if "team_url" in d.columns:
        d["team_img"] = d["team_url"].apply(lambda u: md_img(fetch_image_as_data_url(u), 22))

    # Columns for display
    display_cols: list[str] = []
    for c in [
        "rank_xD",
        "player_img",
        "short_name",
        "team_img",
        "team",
        "total_min_played",
        "xD_per_90",
        "danger_passes_per_90",
        "xD_efficiency",
        "total_xD",
        "danger_passes",
    ]:
        if c in d.columns:
            display_cols.append(c)

    d_disp = d.loc[:, display_cols].reset_index(drop=True)
    gt = GT(d_disp)

    # Render image columns (as markdown with embedded HTML)
    if hasattr(gt, "fmt_markdown"):
        for c in ["player_img", "team_img"]:
            if c in d_disp.columns:
                gt = gt.fmt_markdown(columns=[c])

    # Number formatting
    if hasattr(gt, "fmt_number"):
        as_0dp = [c for c in ["total_min_played", "danger_passes"] if c in d_disp.columns]
        as_2dp = [c for c in ["total_xD", "xD_per_90", "danger_passes_per_90", "xD_efficiency"] if c in d_disp.columns]
        if as_0dp:
            gt = gt.fmt_number(columns=as_0dp, decimals=0)
        if as_2dp:
            gt = gt.fmt_number(columns=as_2dp, decimals=2)

    # Color domains are based on what you're displaying on this page
    def _domain(col: str) -> tuple[float, float] | None:
        if col not in d_disp.columns:
            return None
        s = pd.to_numeric(d_disp[col], errors="coerce")
        if s.notna().sum() == 0:
            return None
        return (float(s.min()), float(s.max()))

    if hasattr(gt, "data_color"):
        xd_dom = _domain("xD_per_90")
        dp_dom = _domain("danger_passes_per_90")
        eff_dom = _domain("xD_efficiency")

        if xd_dom:
            gt = gt.data_color(columns=["xD_per_90"], palette="Reds", domain=xd_dom, reverse=False)
        if dp_dom:
            gt = gt.data_color(
                columns=["danger_passes_per_90"],
                palette="Greens",
                domain=dp_dom,
                reverse=False,
            )
        if eff_dom:
            gt = gt.data_color(
                columns=["xD_efficiency"],
                palette="YlOrBr",
                domain=eff_dom,
                reverse=False,
            )

    # Bold the #1 row (rank == 1) IF it appears on this page
    if "rank_xD" in d_disp.columns:
        gt = gt.tab_style(
            style=style.text(weight="bold"),
            locations=loc.body(rows=lambda t: t["rank_xD"].eq(1)),
        )

    # Header + column labels
    if hasattr(gt, "tab_header"):
        gt = gt.tab_header(
            title=f"Premier League 2024 — {position}",
            subtitle="Ranked by xDanger per 90 (xD_per_90). Efficiency = total_xD / danger_passes.",
        )

    if hasattr(gt, "cols_label"):
        gt = gt.cols_label(
            rank_xD="Rank",
            player_img="",
            short_name="Name",
            team_img="",
            team="Team",
            total_min_played="Minutes",
            xD_per_90="xD / 90",
            danger_passes_per_90="Danger / 90",
            xD_efficiency="xD / Danger",
            total_xD="Total xD",
            danger_passes="Total Danger",
        )

    return gt


def gt_to_html(gt: GT) -> str:
    """Export GT to raw HTML."""
    if hasattr(gt, "as_raw_html"):
        return gt.as_raw_html()
    if hasattr(gt, "_repr_html_"):
        return gt._repr_html_()  # type: ignore[attr-defined]
    raise AttributeError("Great Tables GT object has no HTML export method available.")


# -----------------------------
# UI
# -----------------------------


st.title("⚽ Wyscout xDanger rankings")
st.caption(
    "Browse precomputed position rankings. "
    "Tables are built with Great Tables and rendered as HTML."
)

root = _rankings_root()

with st.sidebar:
    st.header("Controls")

    #st.write("**Dataset directory**")
    #st.code(str(root), language="text")

    # st.sidebar.markdown("### Data directory")
    # st.sidebar.markdown(
    #     f"<div style='word-wrap: break-word; white-space: normal;'>{root}</div>",
    #     unsafe_allow_html=True,
    # )


    
    # st.write("Set `STREAMLIT_RANKINGS_DIR` to point somewhere else.")

    min_minutes = st.number_input(
        "Minimum minutes played",
        min_value=0,
        value=450,
        step=50,
        help="Keeps the leaderboard from being dominated by tiny samples.",
    )

    page_size = st.selectbox(
        "Rows per page",
        options=[10, 15, 20, 25],
        index=0,
    )

try:
    rankings_df = load_rankings_df(str(root))
    rankings_df = _ensure_required_cols(rankings_df)
except Exception as e:
    st.error(str(e))
    st.stop()

positions = sorted(rankings_df["position"].dropna().unique().tolist())
if not positions:
    # Fallback to directory scan (helps when `position` isn't loaded yet)
    positions = _partition_positions_hive(root)

if not positions:
    st.error(
        "No positions found. Expected a `position` column in the parquet files, "
        "or hive partitions like `position=Winger/`."
    )
    st.stop()

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    position = st.selectbox("Position", options=positions, index=0)
with col_b:
    # total rows for this position after min-minutes
    pos_rows = int(
        rankings_df.loc[
            (rankings_df["position"] == position)
            & (rankings_df["total_min_played"].fillna(0) >= min_minutes)
        ].shape[0]
    )
    max_pages = max(1, (pos_rows + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=max_pages, value=1, step=1)
with col_c:
    st.metric("Rows", pos_rows)

gt = make_position_table(
    rankings_df,
    position=position,
    page=int(page),
    page_size=int(page_size),
    min_minutes=int(min_minutes),
)

dark_table = st.sidebar.toggle("Dark tables", value=True)
html = gt.as_raw_html()

if dark_table:
    st.markdown(f'<div class="gt-dark">{html}</div>', unsafe_allow_html=True)
else:
    st.markdown(html, unsafe_allow_html=True)



#html = gt_to_html(gt)

#st.markdown(f'<div class="gt-dark">{html}</div>', unsafe_allow_html=True)

# A little CSS to keep things visually tidy in Streamlit.
html = (
    "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;}</style>\n"
    + html
)

#components.html(html, height=740, scrolling=True)


# Insertion to make table dark mode-ish
#st.markdown(f'<div class="gt-dark">{html}</div>', unsafe_allow_html=True)


with st.expander("What am I looking at?"):
    st.markdown(
        """
**xD / 90** is *Expected Danger per 90 minutes*: The model estimate of how dangerous a player's
passes are, normalized for playing time.

**Danger / 90** How often they play passes leading to a shot within 15 seconds, per 90.

**xD / Danger** is an efficiency ratio: how much xD for each dangerous pass.

**Colors** highlight just those three columns (Reds, Greens, Yellow/Orange). Palette was chosen to accommodate those with CVD. 
"""
    )
