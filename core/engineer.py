"""
Feature Engineering : création et transformation de features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class FeatureEngineer:
    """Génère de nouvelles features à partir d'un DataFrame."""

    def __init__(self):
        self.transformations: List[Dict[str, Any]] = []
        self.log: List[str] = []

    def add_transformation(self, t: Dict[str, Any]):
        self.transformations.append(t)

    def clear(self):
        self.transformations = []
        self.log = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.log = []
        for t in self.transformations:
            df = self._apply(df, t)
        return df

    def _apply(self, df: pd.DataFrame, t: Dict[str, Any]) -> pd.DataFrame:
        action = t["action"]
        new_col = t.get("new_col", "new_feature")
        col = t.get("column")
        col2 = t.get("column2")

        try:
            if action == "log_transform" and col:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[new_col] = np.log1p(df[col].clip(lower=0))
                    self.log.append(f"[log_transform] '{new_col}' = log1p('{col}')")

            elif action == "normalize" and col:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    min_v = df[col].min()
                    max_v = df[col].max()
                    df[new_col] = (df[col] - min_v) / (max_v - min_v + 1e-9)
                    self.log.append(f"[normalize] '{new_col}' = minmax('{col}')")

            elif action == "standardize" and col:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mu = df[col].mean()
                    sigma = df[col].std()
                    df[new_col] = (df[col] - mu) / (sigma + 1e-9)
                    self.log.append(f"[standardize] '{new_col}' = zscore('{col}')")

            elif action == "binarize" and col:
                threshold = float(t.get("threshold", 0))
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[new_col] = (df[col] > threshold).astype(int)
                    self.log.append(f"[binarize] '{new_col}' = ('{col}' > {threshold})")

            elif action == "ratio" and col and col2:
                if col in df.columns and col2 in df.columns:
                    df[new_col] = df[col] / (df[col2].replace(0, np.nan))
                    self.log.append(f"[ratio] '{new_col}' = '{col}' / '{col2}'")

            elif action == "difference" and col and col2:
                if col in df.columns and col2 in df.columns:
                    df[new_col] = df[col] - df[col2]
                    self.log.append(f"[difference] '{new_col}' = '{col}' - '{col2}'")

            elif action == "product" and col and col2:
                if col in df.columns and col2 in df.columns:
                    df[new_col] = df[col] * df[col2]
                    self.log.append(f"[product] '{new_col}' = '{col}' * '{col2}'")

            elif action == "label_encode" and col:
                if col in df.columns:
                    df[new_col] = df[col].astype("category").cat.codes
                    self.log.append(f"[label_encode] '{new_col}' = label_encode('{col}')")

            elif action == "bin" and col:
                bins = int(t.get("bins", 4))
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    df[new_col] = pd.cut(df[col], bins=bins, labels=False)
                    self.log.append(f"[bin] '{new_col}' = cut('{col}', bins={bins})")

            elif action == "age_from_year" and col:
                if col in df.columns:
                    current_year = pd.Timestamp.now().year
                    df[new_col] = current_year - pd.to_numeric(df[col], errors="coerce")
                    self.log.append(f"[age_from_year] '{new_col}' = {current_year} - '{col}'")

            elif action == "custom_expr":
                expr = t.get("expression", "")
                if expr:
                    df[new_col] = df.eval(expr)
                    self.log.append(f"[custom_expr] '{new_col}' = eval('{expr}')")

        except Exception as e:
            self.log.append(f"[ERREUR] action='{action}' col='{col}' : {e}")

        return df

    def get_log(self) -> List[str]:
        return self.log
