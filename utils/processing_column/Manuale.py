import re
import polars as pl

# Pattern per individuare dati sensibili
PATTERNS = {
    'email': re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    'phone': re.compile(r'^\+?39? ?\d{2,4}[ ]?\d{5,8}$'),
    'fiscal_code': re.compile(r'^[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]$'),
     'credit_card': re.compile(
        r'^(?:4\d{12}(?:\d{3})?|'         # Visa
        r'5[1-5]\d{14}|'                  # MasterCard
        r'3[47]\d{13}|'                   # American Express
        r'3(?:0[0-5]|[68]\d)\d{11}|'      # Diners Club
        r'6(?:011|5\d{2})\d{12}|'         # Discover
        r'(?:2131|1800|35\d{3})\d{11})$'  # JCB
    ),
    'ip_address': re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$'),
    'iban': re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]{1,30}$')

}

def detect_sensitive_columns_polars(df: pl.DataFrame, threshold: float = 0.8) -> set:
    """
    Restituisce l'insieme delle colonne che matchano pattern sensibili per almeno `threshold` dei valori.
    """
    sensitive_cols = set()
    n = df.height
    if n == 0:
        return sensitive_cols
    for col in df.columns:
        series = df[col].cast(pl.Utf8).to_list()
        for regex in PATTERNS.values():
            matches = sum(1 for v in series if v and regex.match(v))
            if matches / n >= threshold:
                sensitive_cols.add(col)
                break
    return sensitive_cols

import pandas as pd
import re
from typing import Set

def detect_sensitive_columns_pandas(df: pd.DataFrame, threshold: float = 0.8,max_samples: int = 100):
    """
    Restituisce i nomi delle colonne sensibili usando un campione dei dati.

    Args:
        df: DataFrame da analizzare.
        threshold: frazione minima di valori matching per considerare la colonna sensibile.
        max_samples: numero massimo di righe da campionare per colonna (velocizza l'analisi).

    Returns:
        Set dei nomi delle colonne sensibili.
    """
    sensitive_cols = []
    if df.empty:
        return sensitive_cols

    for col in df.columns:
        # Consideriamo solo righe non-NaN
        non_null_series = df[col].dropna()
        n = len(non_null_series)
        if n == 0:
            continue

        # Campionamento casuale se la colonna è molto lunga
        if n > max_samples:
            series_sample = non_null_series.sample(n=max_samples, random_state=42).astype(str)
            sample_size = max_samples
        else:
            series_sample = non_null_series.astype(str)
            sample_size = n

        # Controllo dei pattern
        for regex in PATTERNS.values():
            matches = series_sample.str.match(regex).sum()
            if matches / sample_size >= threshold:
                sensitive_cols.append(col)
                break

    return sensitive_cols