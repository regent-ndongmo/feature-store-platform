"""
Tests unitaires et d'intégration du Feature Store Platform.
Lancer avec : python -m pytest tests/ -v
"""
import sys
import os
import time
import json
import urllib.request
import shutil
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.cleaner import DataCleaner
from core.engineer import FeatureEngineer
from store.feature_store import FeatureStore, STORE_DIR


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame de test simulant des véhicules d'occasion."""
    np.random.seed(0)
    return pd.DataFrame({
        "vehicle_id": [str(i) for i in range(50)],
        "year":       list(range(2000, 2020)) * 2 + [None] * 10,
        "price":      np.random.randint(2000, 40000, 50).astype(float),
        "odometer":   np.random.randint(5000, 200000, 50).astype(float),
        "fuel":       np.random.choice(["gas", "diesel", "electric", None], 50),
        "condition":  np.random.choice(["good", "excellent", "fair", None], 50),
    })


@pytest.fixture
def store_clean(tmp_path, monkeypatch):
    """Patch le répertoire du store pour ne pas polluer l'env réel."""
    store_path = str(tmp_path / "store_data")
    os.makedirs(store_path, exist_ok=True)
    monkeypatch.setattr("store.feature_store.STORE_DIR", store_path)
    monkeypatch.setattr(
        "store.feature_store.DB_PATH",
        str(tmp_path / "feature_store.db")
    )
    return FeatureStore()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataCleaner:

    def test_drop_duplicates(self, sample_df):
        df = pd.concat([sample_df, sample_df]).reset_index(drop=True)
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "drop_duplicates"})
        result = cleaner.apply(df)
        assert len(result) == len(sample_df)
        assert any("drop_duplicates" in line for line in cleaner.get_log())

    def test_fill_mean(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mean", "column": "year"})
        result = cleaner.apply(sample_df)
        assert result["year"].isna().sum() == 0
        assert any("fill_mean" in line for line in cleaner.get_log())

    def test_fill_median(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_median", "column": "odometer"})
        result = cleaner.apply(sample_df)
        assert result["odometer"].isna().sum() == 0

    def test_fill_mode(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mode", "column": "fuel"})
        result = cleaner.apply(sample_df)
        assert result["fuel"].isna().sum() == 0

    def test_fill_value(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_value", "column": "condition", "value": "unknown"})
        result = cleaner.apply(sample_df)
        assert (result["condition"] == "unknown").any() or result["condition"].isna().sum() == 0

    def test_drop_na_rows(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "drop_na_rows", "column": "year"})
        result = cleaner.apply(sample_df)
        assert result["year"].isna().sum() == 0

    def test_clip_outliers(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "clip_outliers", "column": "price"})
        result = cleaner.apply(sample_df)
        low = sample_df["price"].quantile(0.01)
        high = sample_df["price"].quantile(0.99)
        assert result["price"].min() >= low - 1
        assert result["price"].max() <= high + 1

    def test_drop_column(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "drop_column", "column": "fuel"})
        result = cleaner.apply(sample_df)
        assert "fuel" not in result.columns

    def test_cast_numeric(self):
        df = pd.DataFrame({"val": ["1", "2", "abc", "4"]})
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "cast_numeric", "column": "val"})
        result = cleaner.apply(df)
        assert pd.api.types.is_numeric_dtype(result["val"])

    def test_lowercase_str(self):
        df = pd.DataFrame({"name": ["  Toyota ", "FORD", "Honda"]})
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "lowercase_str", "column": "name"})
        result = cleaner.apply(df)
        assert all(v == v.lower().strip() for v in result["name"])

    def test_chained_rules(self, sample_df):
        """Test chaînage de plusieurs règles."""
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mean", "column": "year"})
        cleaner.add_rule({"action": "fill_median", "column": "odometer"})
        cleaner.add_rule({"action": "drop_duplicates"})
        result = cleaner.apply(sample_df)
        assert result["year"].isna().sum() == 0
        assert result["odometer"].isna().sum() == 0
        assert len(cleaner.get_log()) == 3

    def test_clear_rules(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mean", "column": "year"})
        cleaner.clear_rules()
        assert len(cleaner.rules) == 0
        result = cleaner.apply(sample_df)
        # Sans règle, le df doit être identique
        assert result.equals(sample_df)

    def test_unknown_action_does_not_crash(self, sample_df):
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "unknown_action_xyz", "column": "year"})
        result = cleaner.apply(sample_df)
        assert result is not None  # ne doit pas lever d'exception


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS ENGINEER
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:

    def test_log_transform(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
        result = eng.apply(sample_df)
        assert "log_price" in result.columns
        assert result["log_price"].isna().sum() == 0

    def test_normalize(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
        result = eng.apply(sample_df)
        assert "norm_odo" in result.columns
        assert result["norm_odo"].min() >= 0 - 1e-6
        assert result["norm_odo"].max() <= 1 + 1e-6

    def test_standardize(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "standardize", "column": "price", "new_col": "std_price"})
        result = eng.apply(sample_df)
        assert "std_price" in result.columns
        assert abs(result["std_price"].mean()) < 0.1

    def test_binarize(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({
            "action": "binarize", "column": "price",
            "new_col": "expensive", "threshold": "15000"
        })
        result = eng.apply(sample_df)
        assert set(result["expensive"].dropna().unique()).issubset({0, 1})

    def test_label_encode(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "label_encode", "column": "fuel", "new_col": "fuel_enc"})
        result = eng.apply(sample_df)
        assert "fuel_enc" in result.columns
        assert pd.api.types.is_integer_dtype(result["fuel_enc"])

    def test_ratio(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({
            "action": "ratio", "column": "price",
            "column2": "odometer", "new_col": "price_per_km"
        })
        result = eng.apply(sample_df)
        assert "price_per_km" in result.columns

    def test_difference(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({
            "action": "difference", "column": "price",
            "column2": "odometer", "new_col": "diff"
        })
        result = eng.apply(sample_df)
        assert "diff" in result.columns

    def test_age_from_year(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "age_from_year", "column": "year", "new_col": "age"})
        result = eng.apply(sample_df)
        assert "age" in result.columns
        current_year = pd.Timestamp.now().year
        valid = result["age"].dropna()
        assert (valid > 0).all()
        assert (valid < 100).all()

    def test_bin(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "bin", "column": "price", "new_col": "price_bin", "bins": "4"})
        result = eng.apply(sample_df)
        assert "price_bin" in result.columns

    def test_chained_transformations(self, sample_df):
        eng = FeatureEngineer()
        eng.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
        eng.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
        eng.add_transformation({"action": "label_encode", "column": "fuel", "new_col": "fuel_enc"})
        result = eng.apply(sample_df)
        assert all(c in result.columns for c in ["log_price", "norm_odo", "fuel_enc"])
        assert len(eng.get_log()) == 3

    def test_original_df_not_mutated(self, sample_df):
        original_cols = list(sample_df.columns)
        eng = FeatureEngineer()
        eng.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
        eng.apply(sample_df)
        assert list(sample_df.columns) == original_cols


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS FEATURE STORE
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureStore:

    def test_save_and_list(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="test_group",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price", "odometer"],
            description="Test group",
        )
        groups = store_clean.list_feature_groups()
        names = [g["name"] for g in groups]
        assert "test_group" in names

    def test_get_features_all(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_all",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price", "odometer", "fuel"],
        )
        result = store_clean.get_features("grp_all")
        assert "price" in result.columns
        assert "odometer" in result.columns
        assert len(result) == len(sample_df)

    def test_get_features_subset(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_sub",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price", "odometer", "fuel"],
        )
        result = store_clean.get_features("grp_sub", features=["price"])
        assert "price" in result.columns
        assert "odometer" not in result.columns

    def test_get_features_by_entity_id(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_entity",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price", "odometer"],
        )
        result = store_clean.get_features("grp_entity", entity_ids=["0", "1", "2"])
        assert len(result) == 3

    def test_get_metadata(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_meta",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price"],
            description="Meta test",
        )
        meta = store_clean.get_metadata("grp_meta")
        assert meta is not None
        assert meta["name"] == "grp_meta"
        assert meta["description"] == "Meta test"
        assert "price" in meta["feature_cols"]
        assert meta["version"] == 1

    def test_versioning(self, sample_df, store_clean):
        for _ in range(3):
            store_clean.save_feature_group(
                name="grp_version",
                df=sample_df,
                entity_col="vehicle_id",
                feature_cols=["price"],
            )
        meta = store_clean.get_metadata("grp_version")
        assert meta["version"] == 3

    def test_delete_feature_group(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_del",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price"],
        )
        ok = store_clean.delete_feature_group("grp_del")
        assert ok is True
        groups = store_clean.list_feature_groups()
        assert all(g["name"] != "grp_del" for g in groups)

    def test_delete_nonexistent(self, store_clean):
        ok = store_clean.delete_feature_group("does_not_exist")
        assert ok is False

    def test_get_features_nonexistent(self, store_clean):
        with pytest.raises(ValueError, match="introuvable"):
            store_clean.get_features("nonexistent_group")

    def test_event_timestamp_added(self, sample_df, store_clean):
        store_clean.save_feature_group(
            name="grp_ts",
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price"],
        )
        result = store_clean.get_features("grp_ts")
        assert "event_timestamp" in result.columns

    def test_multiple_groups(self, sample_df, store_clean):
        for name in ["alpha", "beta", "gamma"]:
            store_clean.save_feature_group(
                name=name, df=sample_df,
                entity_col="vehicle_id", feature_cols=["price"],
            )
        groups = store_clean.list_feature_groups()
        names = {g["name"] for g in groups}
        assert {"alpha", "beta", "gamma"}.issubset(names)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS API (INTÉGRATION)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPI:
    """Tests d'intégration de l'API HTTP — nécessite que le serveur soit démarré."""

    API_URL = "http://localhost:5050"

    @pytest.fixture(autouse=True)
    def start_server(self):
        """Démarre l'API avant les tests si elle n'est pas déjà active."""
        from api.server import start_api
        start_api()
        time.sleep(0.5)

    def _get(self, path):
        url = f"{self.API_URL}{path}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()), r.status

    def _post(self, path, body):
        payload = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self.API_URL}{path}",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()), r.status

    def _delete(self, path):
        req = urllib.request.Request(
            f"{self.API_URL}{path}", method="DELETE"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()), r.status

    def test_health(self):
        data, status = self._get("/health")
        assert status == 200
        assert data["status"] == "ok"

    def test_list_feature_groups(self):
        data, status = self._get("/api/feature-groups")
        assert status == 200
        assert "feature_groups" in data
        assert isinstance(data["feature_groups"], list)

    def test_fetch_nonexistent_group(self):
        try:
            self._post("/api/feature-groups/THIS_DOES_NOT_EXIST/fetch", {})
            assert False, "Devrait retourner 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_get_metadata_nonexistent(self):
        try:
            self._get("/api/feature-groups/THIS_DOES_NOT_EXIST")
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_full_workflow_via_api(self, sample_df):
        """Test complet : save via Store → fetch via API."""
        store = FeatureStore()
        group_name = f"api_test_{int(time.time())}"
        store.save_feature_group(
            name=group_name,
            df=sample_df,
            entity_col="vehicle_id",
            feature_cols=["price", "odometer"],
        )

        # Liste
        data, status = self._get("/api/feature-groups")
        assert status == 200
        names = [g["name"] for g in data["feature_groups"]]
        assert group_name in names

        # Métadonnées
        data, status = self._get(f"/api/feature-groups/{group_name}")
        assert status == 200
        assert data["name"] == group_name
        assert "price" in data["feature_cols"]

        # Fetch
        data, status = self._post(
            f"/api/feature-groups/{group_name}/fetch",
            {"features": ["price"], "entity_ids": ["0", "1", "2"]}
        )
        assert status == 200
        assert data["rows"] == 3
        assert "price" in data["columns"]

        # Nettoyage
        store.delete_feature_group(group_name)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS PIPELINE END-TO-END
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Test du pipeline complet : clean → engineer → store → retrieve."""

    def test_full_pipeline(self, sample_df, store_clean):
        # 1. Nettoyage
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mean", "column": "year"})
        cleaner.add_rule({"action": "fill_median", "column": "odometer"})
        cleaner.add_rule({"action": "fill_mode", "column": "fuel"})
        cleaner.add_rule({"action": "drop_duplicates"})
        df_clean = cleaner.apply(sample_df)

        assert df_clean["year"].isna().sum() == 0
        assert df_clean["odometer"].isna().sum() == 0

        # 2. Feature Engineering
        eng = FeatureEngineer()
        eng.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
        eng.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
        eng.add_transformation({"action": "age_from_year", "column": "year", "new_col": "vehicle_age"})
        df_eng = eng.apply(df_clean)

        assert "log_price" in df_eng.columns
        assert "norm_odo" in df_eng.columns
        assert "vehicle_age" in df_eng.columns

        # 3. Sauvegarde dans le store
        features = ["log_price", "norm_odo", "vehicle_age", "fuel"]
        store_clean.save_feature_group(
            name="e2e_test",
            df=df_eng,
            entity_col="vehicle_id",
            feature_cols=features,
            description="Pipeline end-to-end test",
        )

        # 4. Récupération
        retrieved = store_clean.get_features("e2e_test", features=["log_price", "norm_odo"])
        assert "log_price" in retrieved.columns
        assert "norm_odo" in retrieved.columns
        assert len(retrieved) == len(df_eng)
        assert retrieved["norm_odo"].between(0, 1).all()

        # 5. Vérification metadata
        meta = store_clean.get_metadata("e2e_test")
        assert meta["description"] == "Pipeline end-to-end test"
        assert meta["entity_col"] == "vehicle_id"
