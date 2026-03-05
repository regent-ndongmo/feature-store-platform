"""
Suite de tests complète — Feature Store Platform.
Fonctionne SANS pytest : python tests/run_tests.py

Si pytest est installé : python -m pytest tests/run_tests.py -v
"""
import sys
import os
import time
import json
import urllib.request
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import store.feature_store as fs_mod
from core.cleaner import DataCleaner
from core.engineer import FeatureEngineer
from store.feature_store import FeatureStore


# ─── Mini framework de test ────────────────────────────────────────────────────

_passed = []
_failed = []


def test(name, fn):
    try:
        fn()
        _passed.append(name)
        print(f"  ✅  {name}")
    except Exception as e:
        _failed.append((name, str(e)))
        print(f"  ❌  {name}")
        print(f"       {type(e).__name__}: {e}")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def assert_eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg} | attendu={b}, obtenu={a}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "Condition fausse")


# ─── Fixture ──────────────────────────────────────────────────────────────────

def make_df(n=50):
    np.random.seed(42)
    return pd.DataFrame({
        "vehicle_id": [str(i) for i in range(n)],
        "year":       list(range(2000, 2020)) * (n // 20) + [None] * (n % 20),
        "price":      np.random.randint(2000, 40000, n).astype(float),
        "odometer":   np.random.randint(5000, 200000, n).astype(float),
        "fuel":       np.random.choice(["gas", "diesel", "electric", None], n),
        "condition":  np.random.choice(["good", "excellent", "fair", None], n),
    })


# ══════════════════════════════════════════════════════════════════════════════
# TESTS CLEANER
# ══════════════════════════════════════════════════════════════════════════════

section("🧹  DataCleaner")

def t_drop_duplicates():
    df = make_df()
    df = pd.concat([df, df]).reset_index(drop=True)
    c = DataCleaner()
    c.add_rule({"action": "drop_duplicates"})
    r = c.apply(df)
    assert_eq(len(r), 50, "nb lignes après dédoublonnage")
    assert_true(any("drop_duplicates" in l for l in c.get_log()))

test("drop_duplicates", t_drop_duplicates)


def t_fill_mean():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_mean", "column": "year"})
    r = c.apply(df)
    assert_eq(int(r["year"].isna().sum()), 0, "NaN après fill_mean")

test("fill_mean", t_fill_mean)


def t_fill_median():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_median", "column": "odometer"})
    r = c.apply(df)
    assert_eq(int(r["odometer"].isna().sum()), 0)

test("fill_median", t_fill_median)


def t_fill_mode():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_mode", "column": "fuel"})
    r = c.apply(df)
    assert_eq(int(r["fuel"].isna().sum()), 0)

test("fill_mode", t_fill_mode)


def t_fill_value():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_value", "column": "condition", "value": "unknown"})
    r = c.apply(df)
    assert_eq(int(r["condition"].isna().sum()), 0)

test("fill_value", t_fill_value)


def t_drop_na_rows():
    df = make_df()
    before_nan = df["year"].isna().sum()
    c = DataCleaner()
    c.add_rule({"action": "drop_na_rows", "column": "year"})
    r = c.apply(df)
    assert_eq(int(r["year"].isna().sum()), 0)
    assert_true(len(r) < len(df))

test("drop_na_rows", t_drop_na_rows)


def t_clip_outliers():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "clip_outliers", "column": "price"})
    r = c.apply(df)
    low = df["price"].quantile(0.01)
    high = df["price"].quantile(0.99)
    assert_true(r["price"].min() >= low - 1)
    assert_true(r["price"].max() <= high + 1)

test("clip_outliers", t_clip_outliers)


def t_drop_column():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "drop_column", "column": "fuel"})
    r = c.apply(df)
    assert_true("fuel" not in r.columns)

test("drop_column", t_drop_column)


def t_cast_numeric():
    df = pd.DataFrame({"val": ["1", "2", "abc", "4"]})
    c = DataCleaner()
    c.add_rule({"action": "cast_numeric", "column": "val"})
    r = c.apply(df)
    assert_true(pd.api.types.is_numeric_dtype(r["val"]))

test("cast_numeric", t_cast_numeric)


def t_lowercase_str():
    df = pd.DataFrame({"name": ["  Toyota ", "FORD", "Honda"]})
    c = DataCleaner()
    c.add_rule({"action": "lowercase_str", "column": "name"})
    r = c.apply(df)
    assert_true(all(v == v.lower().strip() for v in r["name"]))

test("lowercase_str", t_lowercase_str)


def t_chain_rules():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_mean", "column": "year"})
    c.add_rule({"action": "fill_median", "column": "odometer"})
    c.add_rule({"action": "drop_duplicates"})
    r = c.apply(df)
    assert_eq(int(r["year"].isna().sum()), 0)
    assert_eq(int(r["odometer"].isna().sum()), 0)
    assert_eq(len(c.get_log()), 3)

test("chain_rules", t_chain_rules)


def t_clear_rules():
    df = make_df()
    c = DataCleaner()
    c.add_rule({"action": "fill_mean", "column": "year"})
    c.clear_rules()
    assert_eq(len(c.rules), 0)

test("clear_rules", t_clear_rules)


def t_original_not_mutated_cleaner():
    df = make_df()
    original_cols = list(df.columns)
    c = DataCleaner()
    c.add_rule({"action": "drop_column", "column": "fuel"})
    c.apply(df)
    assert_eq(list(df.columns), original_cols)

test("original_not_mutated", t_original_not_mutated_cleaner)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS ENGINEER
# ══════════════════════════════════════════════════════════════════════════════

section("⚙️   FeatureEngineer")

def t_log_transform():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
    r = e.apply(df)
    assert_true("log_price" in r.columns)
    assert_eq(int(r["log_price"].isna().sum()), 0)

test("log_transform", t_log_transform)


def t_normalize():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
    r = e.apply(df)
    assert_true("norm_odo" in r.columns)
    assert_true(r["norm_odo"].min() >= -1e-6)
    assert_true(r["norm_odo"].max() <= 1 + 1e-6)

test("normalize", t_normalize)


def t_standardize():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "standardize", "column": "price", "new_col": "std_price"})
    r = e.apply(df)
    assert_true("std_price" in r.columns)
    assert_true(abs(r["std_price"].mean()) < 0.1)

test("standardize", t_standardize)


def t_binarize():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "binarize", "column": "price", "new_col": "expensive", "threshold": "15000"})
    r = e.apply(df)
    assert_true(set(r["expensive"].dropna().unique()).issubset({0, 1}))

test("binarize", t_binarize)


def t_label_encode():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "label_encode", "column": "fuel", "new_col": "fuel_enc"})
    r = e.apply(df)
    assert_true("fuel_enc" in r.columns)
    assert_true(pd.api.types.is_integer_dtype(r["fuel_enc"]))

test("label_encode", t_label_encode)


def t_ratio():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "ratio", "column": "price", "column2": "odometer", "new_col": "price_per_km"})
    r = e.apply(df)
    assert_true("price_per_km" in r.columns)

test("ratio", t_ratio)


def t_difference():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "difference", "column": "price", "column2": "odometer", "new_col": "diff"})
    r = e.apply(df)
    assert_true("diff" in r.columns)

test("difference", t_difference)


def t_age_from_year():
    df = make_df()
    # fill NaN first
    df["year"] = df["year"].fillna(2010)
    e = FeatureEngineer()
    e.add_transformation({"action": "age_from_year", "column": "year", "new_col": "age"})
    r = e.apply(df)
    assert_true("age" in r.columns)
    valid = r["age"].dropna()
    assert_true((valid > 0).all())
    assert_true((valid < 100).all())

test("age_from_year", t_age_from_year)


def t_bin():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "bin", "column": "price", "new_col": "price_bin", "bins": "4"})
    r = e.apply(df)
    assert_true("price_bin" in r.columns)

test("bin", t_bin)


def t_chain_fe():
    df = make_df()
    e = FeatureEngineer()
    e.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
    e.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
    e.add_transformation({"action": "label_encode", "column": "fuel", "new_col": "fuel_enc"})
    r = e.apply(df)
    assert_true(all(c in r.columns for c in ["log_price", "norm_odo", "fuel_enc"]))
    assert_eq(len(e.get_log()), 3)

test("chain_transformations", t_chain_fe)


def t_original_not_mutated_fe():
    df = make_df()
    original_cols = list(df.columns)
    e = FeatureEngineer()
    e.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
    e.apply(df)
    assert_eq(list(df.columns), original_cols)

test("original_not_mutated", t_original_not_mutated_fe)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS FEATURE STORE
# ══════════════════════════════════════════════════════════════════════════════

section("🗄️   FeatureStore")

def with_tmp_store(fn):
    """Exécute fn(store) dans un répertoire temporaire isolé."""
    with tempfile.TemporaryDirectory() as tmp:
        orig_dir = fs_mod.STORE_DIR
        orig_db  = fs_mod.DB_PATH
        fs_mod.STORE_DIR = tmp
        fs_mod.DB_PATH   = os.path.join(tmp, "fs.db")
        try:
            fn(FeatureStore())
        finally:
            fs_mod.STORE_DIR = orig_dir
            fs_mod.DB_PATH   = orig_db


def t_save_and_list():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp1", df, "vehicle_id", ["price", "odometer"])
        groups = store.list_feature_groups()
        assert_true(any(g["name"] == "grp1" for g in groups))
    with_tmp_store(fn)

test("save_and_list", t_save_and_list)


def t_get_features_all():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_all", df, "vehicle_id", ["price", "odometer", "fuel"])
        r = store.get_features("grp_all")
        assert_true("price" in r.columns)
        assert_eq(len(r), len(df))
    with_tmp_store(fn)

test("get_features_all", t_get_features_all)


def t_get_features_subset():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_sub", df, "vehicle_id", ["price", "odometer", "fuel"])
        r = store.get_features("grp_sub", features=["price"])
        assert_true("price" in r.columns)
        assert_true("odometer" not in r.columns)
    with_tmp_store(fn)

test("get_features_subset", t_get_features_subset)


def t_get_by_entity_id():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_ent", df, "vehicle_id", ["price"])
        r = store.get_features("grp_ent", entity_ids=["0", "1", "2"])
        assert_eq(len(r), 3)
    with_tmp_store(fn)

test("get_by_entity_id", t_get_by_entity_id)


def t_metadata():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_meta", df, "vehicle_id", ["price"],
                                  description="Test meta")
        meta = store.get_metadata("grp_meta")
        assert_true(meta is not None)
        assert_eq(meta["name"], "grp_meta")
        assert_eq(meta["description"], "Test meta")
        assert_true("price" in meta["feature_cols"])
        assert_eq(meta["version"], 1)
    with_tmp_store(fn)

test("metadata", t_metadata)


def t_versioning():
    def fn(store):
        df = make_df()
        for _ in range(3):
            store.save_feature_group("grp_ver", df, "vehicle_id", ["price"])
        meta = store.get_metadata("grp_ver")
        assert_eq(meta["version"], 3)
    with_tmp_store(fn)

test("versioning", t_versioning)


def t_delete():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_del", df, "vehicle_id", ["price"])
        ok = store.delete_feature_group("grp_del")
        assert_true(ok)
        groups = store.list_feature_groups()
        assert_true(all(g["name"] != "grp_del" for g in groups))
    with_tmp_store(fn)

test("delete", t_delete)


def t_delete_nonexistent():
    def fn(store):
        ok = store.delete_feature_group("does_not_exist")
        assert_true(ok is False)
    with_tmp_store(fn)

test("delete_nonexistent", t_delete_nonexistent)


def t_get_nonexistent():
    def fn(store):
        try:
            store.get_features("no_such_group")
            assert False, "Doit lever ValueError"
        except ValueError as e:
            assert_true("introuvable" in str(e))
    with_tmp_store(fn)

test("get_nonexistent_raises", t_get_nonexistent)


def t_event_timestamp():
    def fn(store):
        df = make_df()
        store.save_feature_group("grp_ts", df, "vehicle_id", ["price"])
        r = store.get_features("grp_ts")
        assert_true("event_timestamp" in r.columns)
    with_tmp_store(fn)

test("event_timestamp_added", t_event_timestamp)


def t_multiple_groups():
    def fn(store):
        df = make_df()
        for name in ["alpha", "beta", "gamma"]:
            store.save_feature_group(name, df, "vehicle_id", ["price"])
        groups = store.list_feature_groups()
        names = {g["name"] for g in groups}
        assert_true({"alpha", "beta", "gamma"}.issubset(names))
    with_tmp_store(fn)

test("multiple_groups", t_multiple_groups)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS API
# ══════════════════════════════════════════════════════════════════════════════

section("🌐  API REST (http://localhost:5050)")

def _api_get(path):
    url = f"http://localhost:5050{path}"
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read()), r.status

def _api_post(path, body):
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:5050{path}", data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read()), r.status


from api.server import start_api
start_api()
time.sleep(0.6)


def t_api_health():
    data, status = _api_get("/health")
    assert_eq(status, 200)
    assert_eq(data["status"], "ok")

test("api_health", t_api_health)


def t_api_list():
    data, status = _api_get("/api/feature-groups")
    assert_eq(status, 200)
    assert_true("feature_groups" in data)
    assert_true(isinstance(data["feature_groups"], list))

test("api_list_feature_groups", t_api_list)


def t_api_fetch_nonexistent():
    try:
        _api_post("/api/feature-groups/THIS_DOES_NOT_EXIST/fetch", {})
        assert False, "Devrait retourner 404"
    except urllib.error.HTTPError as e:
        assert_eq(e.code, 404)

test("api_fetch_404", t_api_fetch_nonexistent)


def t_api_metadata_nonexistent():
    try:
        _api_get("/api/feature-groups/THIS_DOES_NOT_EXIST")
        assert False
    except urllib.error.HTTPError as e:
        assert_eq(e.code, 404)

test("api_metadata_404", t_api_metadata_nonexistent)


def t_api_full_workflow():
    """Save → list via API → fetch via API."""
    df = make_df()
    group_name = f"api_test_{int(time.time())}"
    store = FeatureStore()
    store.save_feature_group(group_name, df, "vehicle_id", ["price", "odometer"],
                              description="API workflow test")

    data, status = _api_get("/api/feature-groups")
    assert_eq(status, 200)
    names = [g["name"] for g in data["feature_groups"]]
    assert_true(group_name in names)

    data, status = _api_get(f"/api/feature-groups/{group_name}")
    assert_eq(status, 200)
    assert_eq(data["name"], group_name)

    data, status = _api_post(
        f"/api/feature-groups/{group_name}/fetch",
        {"features": ["price"], "entity_ids": ["0", "1", "2"]}
    )
    assert_eq(status, 200)
    assert_eq(data["rows"], 3)
    assert_true("price" in data["columns"])

    store.delete_feature_group(group_name)

test("api_full_workflow", t_api_full_workflow)


# ══════════════════════════════════════════════════════════════════════════════
# TEST END-TO-END
# ══════════════════════════════════════════════════════════════════════════════

section("🔄  Pipeline End-to-End")


def t_e2e():
    def fn(store):
        df = make_df()

        # 1. Nettoyage
        cleaner = DataCleaner()
        cleaner.add_rule({"action": "fill_mean", "column": "year"})
        cleaner.add_rule({"action": "fill_median", "column": "odometer"})
        cleaner.add_rule({"action": "fill_mode", "column": "fuel"})
        cleaner.add_rule({"action": "drop_duplicates"})
        df_clean = cleaner.apply(df)
        assert_eq(int(df_clean["year"].isna().sum()), 0)

        # 2. Feature Engineering
        eng = FeatureEngineer()
        eng.add_transformation({"action": "log_transform", "column": "price", "new_col": "log_price"})
        eng.add_transformation({"action": "normalize", "column": "odometer", "new_col": "norm_odo"})
        eng.add_transformation({"action": "age_from_year", "column": "year", "new_col": "vehicle_age"})
        df_eng = eng.apply(df_clean)
        assert_true(all(c in df_eng.columns for c in ["log_price", "norm_odo", "vehicle_age"]))

        # 3. Store
        features = ["log_price", "norm_odo", "vehicle_age"]
        store.save_feature_group("e2e_test", df_eng, "vehicle_id", features,
                                  description="Pipeline E2E")

        # 4. Retrieve
        retrieved = store.get_features("e2e_test", features=["log_price", "norm_odo"])
        assert_true("log_price" in retrieved.columns)
        assert_eq(len(retrieved), len(df_eng))
        assert_true(retrieved["norm_odo"].between(0, 1).all())

        # 5. Metadata
        meta = store.get_metadata("e2e_test")
        assert_eq(meta["description"], "Pipeline E2E")
        assert_eq(meta["entity_col"], "vehicle_id")

    with_tmp_store(fn)

test("full_pipeline_e2e", t_e2e)


# ─── Rapport final ─────────────────────────────────────────────────────────────

total = len(_passed) + len(_failed)
print(f"\n{'═'*60}")
print(f"  RÉSULTATS  :  {len(_passed)}/{total} tests passés")
print(f"{'═'*60}")
if _failed:
    print(f"\n  ❌ Tests échoués :")
    for name, err in _failed:
        print(f"     • {name} → {err}")
    sys.exit(1)
else:
    print(f"\n  🎉  Tous les tests passent !\n")
