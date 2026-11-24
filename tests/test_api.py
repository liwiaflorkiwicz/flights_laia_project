from fastapi.testclient import TestClient
from src.api.main import app

def test_health_endpoint() -> None:
    """Test the /health endpoint status check and model presence readout."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == "True"


def test_root_redirect() -> None:
    """Verify root URL automatic redirection to interactive OpenAPI documentation."""
    with TestClient(app) as client:
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)


def test_classify_valid_payload() -> None:
    """Validate end-to-end multi-class delay prediction via the /classify endpoint."""
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
        assert response.status_code == 200
        data = response.json()
        assert "classifications" in data


def test_classify_empty_instances() -> None:
    """Ensure HTTP 400 Bad Request is returned when submitting empty instance lists."""
    with TestClient(app) as client:
        response = client.post("/classify", json={"instances": []})
        assert response.status_code == 400