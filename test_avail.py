import urllib.request, json
BASE_URL = "http://localhost:11434"
model = "llama3.1:8b"
try:
    with urllib.request.urlopen(f"{BASE_URL}/api/tags", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name","").strip() for m in payload.get("models", [])]
        print("Models:", models)
        print("Available:", any(model in m for m in models))
except Exception as e:
    print("EXCEPTION:", type(e).__name__, "-", e)