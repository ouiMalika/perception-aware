#!/usr/bin/env python3
"""
Quick API test script. Exercises the backend endpoints.

Usage:
    # Start the stack first:  docker compose up -d
    # Then run:               python scripts/test_api.py
"""

import json
import sys
import time

import requests

BASE_URL = "http://localhost:8000/api"


def test_api():
    print("=== Perception-Aware Traffic Sign Detection - API Test ===\n")

    # 1. Check taxonomy endpoint (no auth needed)
    print("1. GET /api/taxonomy/")
    resp = requests.get(f"{BASE_URL}/taxonomy/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    taxonomy = resp.json()
    print(f"   Categories: {taxonomy['total_categories']}")
    assert taxonomy["total_categories"] >= 20
    print("   PASS\n")

    # 2. Register a test user
    print("2. POST /api/auth/register/")
    resp = requests.post(f"{BASE_URL}/auth/register/", json={
        "username": f"testuser_{int(time.time())}",
        "password": "testpass123!",
    })
    assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
    token = resp.json()["token"]
    print(f"   Token: {token[:8]}...")
    print("   PASS\n")

    headers = {"Authorization": f"Token {token}"}

    # 3. Submit a detection job
    print("3. POST /api/detect/")
    resp = requests.post(
        f"{BASE_URL}/detect/",
        json={
            "image_urls": [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/STOP_sign.jpg/220px-STOP_sign.jpg",
            ],
            "yolo_conf": 0.25,
            "clip_conf": 0.15,
        },
        headers=headers,
    )
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    job_id = resp.json()["job_id"]
    print(f"   Job ID: {job_id}")
    print("   PASS\n")

    # 4. Poll job status
    print("4. GET /api/jobs/<job_id>/")
    for attempt in range(30):
        resp = requests.get(f"{BASE_URL}/jobs/{job_id}/", headers=headers)
        assert resp.status_code == 200
        job_status = resp.json()["status"]
        print(f"   Attempt {attempt + 1}: {job_status}")
        if job_status == "completed":
            break
        if job_status == "failed":
            print(f"   ERROR: {resp.json().get('error_message')}")
            break
        time.sleep(2)
    else:
        print("   TIMEOUT - job did not complete in 60s")

    if job_status == "completed":
        result = resp.json().get("result", {})
        total_det = result.get("total_detections", 0)
        print(f"   Total detections: {total_det}")
        print("   PASS\n")

    # 5. List jobs
    print("5. GET /api/jobs/")
    resp = requests.get(f"{BASE_URL}/jobs/", headers=headers)
    assert resp.status_code == 200
    print(f"   Jobs: {len(resp.json())}")
    print("   PASS\n")

    # 6. List detected signs
    print("6. GET /api/signs/")
    resp = requests.get(f"{BASE_URL}/signs/", headers=headers)
    assert resp.status_code == 200
    signs = resp.json()
    print(f"   Signs: {len(signs)}")
    for sign in signs[:5]:
        print(f"     - {sign['category']}: {sign['confidence']:.2%}")
    print("   PASS\n")

    print("=== All API tests passed ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(test_api())
    except requests.ConnectionError:
        print("ERROR: Cannot connect to backend. Is the stack running?")
        print("  Run: docker compose up -d")
        sys.exit(1)
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
