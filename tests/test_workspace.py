import pytest
from fastapi.testclient import TestClient
from src import store
from src.api.main import app
from api.index import app as serverless_app
from src.privacy import workspace
from src.agent.chat_graph import _run_sql


@pytest.mark.parametrize("target", [app, serverless_app])
def test_visitors_cannot_read_write_delete_or_query_other_runs(engine, monkeypatch, target):
    monkeypatch.setattr(store, "get_engine", lambda *a, **k: engine)
    owner = store.create_source("webcam", model_version="owner", conf_threshold=.4)
    a = TestClient(target, headers={"X-Workspace-Key": "a" * 64})
    b = TestClient(target, headers={"X-Workspace-Key": "b" * 64})
    sid = a.post("/api/v1/client-sessions", json={"kind":"webcam", "model_version":"test", "conf_threshold":.4}).json()["id"]
    det = {"class_id":0,"class_label":"person","confidence":.9,"bbox_x":1,"bbox_y":2,"bbox_w":3,"bbox_h":4}
    payload = {"items":[{"frame_number":0,"ts_seconds":0,"detections":[det]}]}
    assert a.post(f"/api/v1/client-sessions/{sid}/detections", json=payload).status_code == 200
    assert b.get("/api/v1/sources").json() == []
    assert b.get(f"/api/v1/sources/{sid}/detections").json() == []
    assert b.get("/api/v1/stats").json()["totals"]["detections"] == 0
    assert b.post(f"/api/v1/client-sessions/{sid}/detections", json=payload).status_code == 404
    assert b.delete(f"/api/v1/sources/{sid}").status_code == 404
    assert b.delete("/api/v1/sources").json()["sources"] == 0
    import hashlib
    token = workspace.set(hashlib.sha256(b"b" * 64).hexdigest())
    try:
        assert _run_sql("SELECT COUNT(*) AS n FROM detections")[0]["n"] == 0
    finally:
        workspace.reset(token)
    assert len(a.get("/api/v1/sources").json()) == 1
    assert a.delete("/api/v1/sources").json()["sources"] == 1
    assert [r["id"] for r in store.list_sources()] == [owner]


def test_invalid_capability_is_rejected():
    assert TestClient(app).get("/api/v1/sources", headers={"X-Workspace-Key":"guess"}).status_code == 400
