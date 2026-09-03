"""Authentication, role management and health endpoints."""
from __future__ import annotations


def test_health_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_public_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    assert "company_name" in response.json()


def test_login_rejects_bad_password(client):
    response = client.post("/api/auth/login", json={"email": "admin@example.com", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_login_does_not_leak_unknown_accounts(client):
    response = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect email or password"


def test_me_requires_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_profile(client, auth_headers):
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["email"] == "admin@example.com"
    assert response.json()["is_superuser"] is True


def test_role_crud(client, auth_headers):
    created = client.post(
        "/api/roles",
        headers=auth_headers,
        json={
            "title": "Data Analyst",
            "department": "Analytics",
            "description": "SQL and dashboards",
            "questions": [
                {"text": "How do you approach a messy dataset?", "time_limit_seconds": 120},
                {"text": "Explain a dashboard you are proud of.", "time_limit_seconds": 180},
            ],
        },
    )
    assert created.status_code == 201, created.text
    role = created.json()
    assert len(role["questions"]) == 2
    assert role["questions"][0]["order_index"] == 0

    listed = client.get("/api/roles", headers=auth_headers)
    assert listed.status_code == 200
    assert any(item["id"] == role["id"] and item["question_count"] == 2 for item in listed.json())

    patched = client.patch(
        f"/api/roles/{role['id']}", headers=auth_headers, json={"title": "Senior Data Analyst"}
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == "Senior Data Analyst"

    assert client.delete(f"/api/roles/{role['id']}", headers=auth_headers).status_code == 204
    assert client.get(f"/api/roles/{role['id']}", headers=auth_headers).status_code == 404


def test_role_requires_questions(client, auth_headers):
    response = client.post(
        "/api/roles", headers=auth_headers, json={"title": "Empty Role", "questions": []}
    )
    assert response.status_code == 422


def test_roles_require_auth(client):
    assert client.get("/api/roles").status_code == 401
