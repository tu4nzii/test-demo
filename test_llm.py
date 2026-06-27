"""Test LLM API connectivity and basic functionality."""
import sys, time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from model_api_config import get_chat_completion_url, get_headers, get_model_name
import requests

URL = get_chat_completion_url()
HEADERS = get_headers()
MODEL = get_model_name()

print(f"URL: {URL}")
print(f"Model: {MODEL}")
print()

# Test 1: Basic connectivity (list models)
print("Test 1: Connectivity...")
models_url = URL.replace("/chat/completions", "/models")
t0 = time.time()
try:
    r = requests.get(models_url, headers=HEADERS, timeout=10)
    elapsed = time.time() - t0
    print(f"  OK ({elapsed:.1f}s) status={r.status_code}")
    if r.ok:
        data = r.json()
        if "data" in data:
            names = [m["id"] for m in data["data"][:5]]
            print(f"  Available models: {names}...")
except Exception as e:
    print(f"  FAIL ({time.time()-t0:.1f}s): {e}")

# Test 2: Simple text chat
print("\nTest 2: Simple text chat...")
payload = {
    "model": MODEL,
    "temperature": 0.1,
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
}
t0 = time.time()
try:
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=15)
    elapsed = time.time() - t0
    print(f"  OK ({elapsed:.1f}s) status={r.status_code}")
    if r.ok:
        answer = r.json()["choices"][0]["message"]["content"]
        print(f"  Response: '{answer.strip()}'")
    else:
        print(f"  Error: {r.text[:200]}")
except Exception as e:
    print(f"  FAIL ({time.time()-t0:.1f}s): {e}")

# Test 3: Vision (encode a tiny white image as base64)
print("\nTest 3: Vision (white 50x50 image)...")
import base64, numpy as np
try:
    import cv2
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("utf-8")
except ImportError:
    # Fallback: minimal PNG in base64
    b64 = "iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAH0lEQVR4nO3BAQ0AAADCoPdPbQ8HFAAAAAAAAAAAAPBuQWAAAcT3T3oAAAAASUVORK5CYII="

payload = {
    "model": MODEL,
    "temperature": 0.1,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "What color is this image? Reply with one word."},
        ],
    }],
}
t0 = time.time()
try:
    r = requests.post(URL, headers=HEADERS, json=payload, timeout=15)
    elapsed = time.time() - t0
    print(f"  OK ({elapsed:.1f}s) status={r.status_code}")
    if r.ok:
        answer = r.json()["choices"][0]["message"]["content"]
        print(f"  Response: '{answer.strip()}'")
    else:
        print(f"  Error: {r.text[:200]}")
except Exception as e:
    print(f"  FAIL ({time.time()-t0:.1f}s): {e}")

print("\nDone.")
