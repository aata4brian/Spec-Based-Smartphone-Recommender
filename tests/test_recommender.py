import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import fuzzy_smartphone_mamdani as engine

client = TestClient(engine.app)


def test_storage_does_not_invent_capacity_from_ram():
    assert np.isnan(engine.extract_storage_gb(pd.Series({"Model Name": "Phone", "RAM": "12GB"})))
    assert engine.extract_storage_gb(pd.Series({"Model Name": "Phone 1TB", "RAM": "12GB"})) == 1024
    assert engine.extract_storage_gb(pd.Series({"Model Name": "Phone 1TB", "Storage_GB": 256})) == 256


@pytest.mark.parametrize("label", ["low", "medium", "high"])
def test_cached_memberships_preserve_original_definition(label):
    np.testing.assert_array_equal(
        engine.cached_output_membership(label),
        engine.output_membership(np.linspace(0, 100, 1001), label),
    )
    assert not engine.cached_output_membership(label).flags.writeable


@pytest.mark.parametrize("payload", [
    {"budget": "invalid"}, {"priority": ["unknown"]}, {"min_storage": -1},
])
def test_invalid_preferences_rejected(payload):
    assert client.post("/recommend", json=payload).status_code == 422


@pytest.mark.parametrize("budget", ["low", "medium", "high"])
def test_real_dataset_recommendations_are_ranked_and_respect_storage(budget):
    response = client.post("/recommend", json={"budget": budget, "priority": ["Camera", "Battery"], "min_storage": 256})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "OK"
    rows = result["recommendations"]
    assert 1 <= len(rows) <= 20
    assert [r["Rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert [r["Score"] for r in rows] == sorted([r["Score"] for r in rows], reverse=True)
    assert all(r["Storage"] >= 256 and 0 <= r["Score"] <= 100 for r in rows)
    assert all(np.isfinite(v) for r in rows for v in r["Radar"].values())


def test_no_candidate_is_a_valid_empty_result():
    result = client.post("/recommend", json={"min_storage": 1000000}).json()
    assert result["status"] == "EMPTY"
    assert result["recommendations"] == []


def test_soft_budget_filter_is_explicitly_characterized():
    frame = pd.DataFrame({"price_usd": [200, 900]})
    assert len(engine.apply_budget_filter(frame, "low")) == 2


def test_duplicate_priorities_do_not_repeat_rules():
    request = engine.RecommendationRequest(priority=["RAM", " ram "])
    assert request.priority == ["ram"]


def test_single_process_frontend_and_dataset_health():
    assert client.get("/health").json()["dataset_rows"] > 0
    page = client.get("/ui")
    assert page.status_code == 200
    assert 'id="recommendForm"' in page.text
