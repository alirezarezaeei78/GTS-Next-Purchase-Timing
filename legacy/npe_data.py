"""Historical customer-day adapters for Online Retail II and Ta-Feng.

The CLI wrapper drops missing customer IDs before this adapter by default.
The explicit legacy flag preserves the older pooled-ID behavior for audit.
"""
from __future__ import annotations
import warnings,zipfile
from pathlib import Path
from typing import Dict,Iterable,Optional,Sequence,Tuple
import numpy as np
import pandas as pd


CUSTOMER_CANDIDATES = ["customers_id", "customer_id", "Customer ID", "CustomerID",
                       "customer", "CUSTOMER_ID"]

PRODUCT_CANDIDATES = ["product_id", "articles_id", "StockCode", "PRODUCT_ID",
                      "product_category", "article_id"]

DATE_CANDIDATES = ["d_dat", "InvoiceDate", "Invoice Date", "TRANSACTION_DT",
                   "event_date", "date", "Date"]

TIME_CANDIDATES = ["SaleTime", "time", "Time"]

QUANTITY_CANDIDATES = ["quantity", "Quantity", "tedad", "AMOUNT", "Num"]

AMOUNT_CANDIDATES = ["sale_price_total", "NetAmount", "value [USD]", "basketprice",
                     "price_purchase", "SALES_PRICE", "price", "price_item",
                     "UnitPrice", "Price"]

def _first_existing(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def _read_any(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    """Read csv/xlsx/zip transparently (Online Retail II ships as xlsx/zip)."""
    ext = path.suffix.lower()
    if ext == ".zip":
        with zipfile.ZipFile(path) as zf:
            inner = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
            if not inner:
                raise FileNotFoundError(f"No csv/xlsx inside {path}")
            extracted = Path(zf.extract(inner[0], path=path.parent))
        return _read_any(extracted, max_rows)
    if ext in (".xlsx", ".xls"):
        xl = pd.ExcelFile(path)
        frames = [pd.read_excel(path, sheet_name=s) for s in xl.sheet_names]
        raw = pd.concat(frames, ignore_index=True)
        return raw.head(max_rows) if max_rows else raw
    if ext == ".parquet":
        raw = pd.read_parquet(path)
        return raw.head(max_rows) if max_rows else raw
    return pd.read_csv(path, nrows=max_rows, low_memory=False)

def _parse_dates(df: pd.DataFrame, date_col: str, time_col: Optional[str]) -> pd.Series:
    raw = df[date_col].astype(str)
    has_time = raw.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False).mean() > 0.05
    if time_col and not has_time:
        t = df[time_col]
        usable = t.notna() & (t.astype(str).str.strip() != "")
        if usable.mean() > 0.05:
            raw = raw.where(~usable, raw + " " + t.astype(str))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(raw, errors="coerce", dayfirst=False)
        if parsed.notna().mean() < 0.5:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    return parsed

def _to_basket_events(lines: pd.DataFrame) -> pd.DataFrame:
    """Line items -> one customer-day basket event (leakage guard G1)."""
    return (
        lines.groupby(["customer_id", "event_date"], sort=False)
        .agg(sale_price_total=("amount", "sum"),
             quantity=("quantity", "sum"),
             line_items=("product_id", "size"),
             unique_products_in_event=("product_id", "nunique"),
             product_ids=("product_id",
                          lambda s: tuple(sorted(set(s.astype(str))))))
        .reset_index()
    )

def load_generic(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Adapter for line-item transaction logs (other-source, other-source, Online Retail II).

    Steps: infer columns -> drop cancellation invoices (Online Retail II 'C'
    invoices) -> drop non-positive quantities (returns) -> compute line amount
    (total column if present, else price*quantity) -> basketize by customer-day.
    """
    path = Path(path)
    raw = _read_any(path, max_rows)
    for inv_col in ["Invoice", "InvoiceNo"]:
        if inv_col in raw.columns:
            raw = raw[~raw[inv_col].astype(str).str.startswith("C")]

    customer = _first_existing(raw.columns, CUSTOMER_CANDIDATES)
    product = _first_existing(raw.columns, PRODUCT_CANDIDATES)
    date = _first_existing(raw.columns, DATE_CANDIDATES)
    time_col = _first_existing(raw.columns, TIME_CANDIDATES)
    quantity = _first_existing(raw.columns, QUANTITY_CANDIDATES)
    missing = [n for n, v in [("customer", customer), ("product", product),
                              ("date", date)] if v is None]
    if missing:
        raise ValueError(f"{path.name}: cannot infer {missing}; "
                         f"columns={list(raw.columns)}")

    # Amount: prefer an explicit basket/net total, else unit price * quantity.
    amount = None
    for col in ["sale_price_total", "NetAmount", "value [USD]", "basketprice"]:
        if col in raw.columns:
            amount = pd.to_numeric(raw[col], errors="coerce")
            break
    if amount is None:
        price_col = _first_existing(raw.columns, AMOUNT_CANDIDATES)
        if price_col is not None and quantity is not None and price_col != quantity:
            amount = (pd.to_numeric(raw[price_col], errors="coerce")
                      * pd.to_numeric(raw[quantity], errors="coerce"))
        elif price_col is not None:
            amount = pd.to_numeric(raw[price_col], errors="coerce")
        else:
            amount = pd.Series(1.0, index=raw.index)

    lines = pd.DataFrame({
        "customer_id": raw[customer].astype(str),
        "product_id": raw[product].astype(str),
        "event_ts": _parse_dates(raw, date, time_col),
        "amount": pd.to_numeric(amount, errors="coerce").fillna(0.0),
        "quantity": (pd.to_numeric(raw[quantity], errors="coerce").fillna(1.0)
                     if quantity is not None else 1.0),
    })
    if quantity is not None:
        lines = lines[lines["quantity"] > 0]
    lines = lines.dropna(subset=["customer_id", "product_id", "event_ts"])
    lines["event_date"] = lines["event_ts"].dt.floor("D")
    return _to_basket_events(lines)

def load_ta_feng(path: str | Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Ta-Feng grocery: explicit mapping because its column names collide with
    nothing in the generic candidates (AMOUNT is *quantity* here, not money)."""
    raw = pd.read_csv(path, nrows=max_rows)
    lines = pd.DataFrame({
        "customer_id": raw["CUSTOMER_ID"].astype(str),
        "product_id": raw["PRODUCT_ID"].astype(str),
        "event_ts": pd.to_datetime(raw["TRANSACTION_DT"], errors="coerce"),
        "quantity": pd.to_numeric(raw["AMOUNT"], errors="coerce").fillna(1.0),
        "amount": (pd.to_numeric(raw["SALES_PRICE"], errors="coerce").fillna(0.0)
                   * pd.to_numeric(raw["AMOUNT"], errors="coerce").fillna(1.0)),
    })
    lines = lines[lines["quantity"] > 0].dropna(
        subset=["customer_id", "product_id", "event_ts"])
    lines["event_date"] = lines["event_ts"].dt.floor("D")
    return _to_basket_events(lines)
