from tests.api.support import client


def test_health():
    with client() as api:
        response = api.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
