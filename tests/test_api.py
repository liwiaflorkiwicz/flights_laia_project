from fastapi.testclient import TestClient
from src.api.main import app

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == "True"

def test_root_redirect():
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)

def test_classify_valid_payload():
    payload = {
        "instances": [
            {
                "FL_DATE": "2023-05-15",
                "OP_UNIQUE_CARRIER": "AA",
                "ORIGIN": "JFK",
                "DEST": "LAX",
                "CRS_DEP_TIME": 1400,
                "CRS_ARR_TIME": 1730,
                "CRS_ELAPSED_TIME": 330.0,
                "DISTANCE": 2475.0,
            }
        ]
    }
    with TestClient(app) as client:
        response = client.post("/classify", json=payload)
        if response.status_code == 422:
            print("\n[DEBUG Pydantic 422 Detail]:", response.json())
        assert response.status_code == 200

def test_classify_empty_instances():
    with TestClient(app) as client:
        response = client.post("/classify", json={"instances": []})
        assert response.status_code == 400