"""
test_auth_full.py — Week 4
Tests the full auth flow: register → login → access /me → refresh.
Run with: pytest tests/test_auth_full.py -v
"""
import pytest


@pytest.mark.asyncio
async def test_register_user(client):
    response = await client.post("/api/auth/register", json={
        "email": "test_week4@example.com",
        "username": "week4tester",
        "password": "SecurePass123",
        "full_name": "Week Four Tester",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test_week4@example.com"
    assert "id" in data


@pytest.mark.asyncio
async def test_register_duplicate_email_fails(client):
    payload = {
        "email": "dup@example.com",
        "username": "dupuser1",
        "password": "SecurePass123",
    }
    await client.post("/api/auth/register", json=payload)
    payload["username"] = "dupuser2"  # different username, same email
    response = await client.post("/api/auth/register", json=payload)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_login_success(client):
    await client.post("/api/auth/register", json={
        "email": "login_test@example.com",
        "username": "loginuser",
        "password": "SecurePass123",
    })
    response = await client.post("/api/auth/login", json={
        "email": "login_test@example.com",
        "password": "SecurePass123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password_fails(client):
    await client.post("/api/auth/register", json={
        "email": "wrongpass@example.com",
        "username": "wrongpassuser",
        "password": "CorrectPass123",
    })
    response = await client.post("/api/auth/login", json={
        "email": "wrongpass@example.com",
        "password": "IncorrectPass999",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_requires_token(client):
    response = await client.get("/api/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_with_valid_token(client):
    await client.post("/api/auth/register", json={
        "email": "me_test@example.com",
        "username": "metestuser",
        "password": "SecurePass123",
    })
    login_resp = await client.post("/api/auth/login", json={
        "email": "me_test@example.com",
        "password": "SecurePass123",
    })
    token = login_resp.json()["access_token"]

    response = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["email"] == "me_test@example.com"


@pytest.mark.asyncio
async def test_refresh_token_flow(client):
    await client.post("/api/auth/register", json={
        "email": "refresh_test@example.com",
        "username": "refreshuser",
        "password": "SecurePass123",
    })
    login_resp = await client.post("/api/auth/login", json={
        "email": "refresh_test@example.com",
        "password": "SecurePass123",
    })
    refresh_token = login_resp.json()["refresh_token"]

    response = await client.post("/api/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == 200
    assert "access_token" in response.json()
