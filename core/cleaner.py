"""
Mécanismes de nettoyage des données.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataCleaner:
    """Applique une suite de règles de nettoyage sur un DataFrame."""

    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.log: List[str] = []

    def add_rule(self, rule: Dict[str, Any]):
        self.rules.append(rule)

    def clear_rules(self):
        self.rules = []
        self.log = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.log = []
        for rule in self.rules:
            df = self._apply_rule(df, rule)
        return df

    def _apply_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.DataFrame:
        action = rule["action"]
        col = rule.get("column")

        if action == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            self.log.append(f"[drop_duplicates] Supprimé {before - len(df)} doublons")

        elif action == "drop_column" and col:
            if col in df.columns:
                df = df.drop(columns=[col])
                self.log.append(f"[drop_column] Colonne '{col}' supprimée")

        elif action == "fill_mean" and col:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                filled = df[col].isna().sum()
                df[col] = df[col].fillna(mean_val)
                self.log.append(f"[fill_mean] '{col}' : {filled} valeurs remplies avec mean={mean_val:.4f}")

        elif action == "fill_median" and col:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                med_val = df[col].median()
                filled = df[col].isna().sum()
                df[col] = df[col].fillna(med_val)
                self.log.append(f"[fill_median] '{col}' : {filled} valeurs remplies avec median={med_val:.4f}")

        elif action == "fill_mode" and col:
            if col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    filled = df[col].isna().sum()
                    df[col] = df[col].fillna(mode_val[0])
                    self.log.append(f"[fill_mode] '{col}' : {filled} valeurs remplies avec mode={mode_val[0]}")

        elif action == "fill_value" and col:
            value = rule.get("value", "")
            if col in df.columns:
                filled = df[col].isna().sum()
                df[col] = df[col].fillna(value)
                self.log.append(f"[fill_value] '{col}' : {filled} valeurs remplies avec '{value}'")

        elif action == "drop_na_rows" and col:
            before = len(df)
            df = df.dropna(subset=[col])
            self.log.append(f"[drop_na_rows] Supprimé {before - len(df)} lignes avec NaN dans '{col}'")

        elif action == "drop_all_na_rows":
            before = len(df)
            df = df.dropna(how="all")
            self.log.append(f"[drop_all_na_rows] Supprimé {before - len(df)} lignes entièrement vides")

        elif action == "clip_outliers" and col:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                low = df[col].quantile(0.01)
                high = df[col].quantile(0.99)
                before_count = ((df[col] < low) | (df[col] > high)).sum()
                df[col] = df[col].clip(low, high)
                self.log.append(f"[clip_outliers] '{col}' : {before_count} valeurs clippées entre [{low:.2f}, {high:.2f}]")

        elif action == "cast_numeric" and col:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                self.log.append(f"[cast_numeric] '{col}' converti en numérique")

        elif action == "lowercase_str" and col:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                self.log.append(f"[lowercase_str] '{col}' converti en minuscules")

        return df

    def get_log(self) -> List[str]:
        return self.log
