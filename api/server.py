"""
API REST locale (Flask) pour récupérer les features du store.
Lance un serveur HTTP sur le port 5050 en arrière-plan.

Endpoints :
  GET  /api/feature-groups               → liste tous les groupes
  GET  /api/feature-groups/<name>        → métadonnées d'un groupe
  POST /api/feature-groups/<name>/fetch  → récupère les données
  DELETE /api/feature-groups/<name>      → supprime un groupe

Exemple de requête fetch :
  POST /api/feature-groups/my_group/fetch
  Body JSON : { "features": ["col1", "col2"], "entity_ids": ["0", "1"] }
"""
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from store.feature_store import FeatureStore

PORT = 5050
_server_thread = None
_httpd = None


class FeatureStoreHandler(BaseHTTPRequestHandler):
    store = FeatureStore()

    def log_message(self, format, *args):
        pass  # silence les logs HTTP dans le terminal

    def _send_json(self, code: int, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        # GET /api/feature-groups
        if parts == ["api", "feature-groups"]:
            groups = self.store.list_feature_groups()
            self._send_json(200, {"feature_groups": groups})

        # GET /api/feature-groups/<name>
        elif len(parts) == 3 and parts[0] == "api" and parts[1] == "feature-groups":
            name = parts[2]
            meta = self.store.get_metadata(name)
            if meta:
                self._send_json(200, meta)
            else:
                self._send_json(404, {"error": f"Groupe '{name}' introuvable"})

        # GET /health
        elif parts == ["health"]:
            self._send_json(200, {"status": "ok", "port": PORT})

        else:
            self._send_json(404, {"error": "Route non trouvée"})

    def do_POST(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        # POST /api/feature-groups/<name>/fetch
        if (
            len(parts) == 4
            and parts[0] == "api"
            and parts[1] == "feature-groups"
            and parts[3] == "fetch"
        ):
            name = parts[2]
            content_length = int(self.headers.get("Content-Length", 0))
            body = {}
            if content_length > 0:
                raw = self.rfile.read(content_length)
                try:
                    body = json.loads(raw)
                except Exception:
                    body = {}

            features = body.get("features")
            entity_ids = body.get("entity_ids")

            try:
                df = self.store.get_features(name, features=features, entity_ids=entity_ids)
                records = df.to_dict(orient="records")
                self._send_json(200, {
                    "group": name,
                    "rows": len(records),
                    "columns": list(df.columns),
                    "data": records,
                })
            except ValueError as e:
                self._send_json(404, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Route non trouvée"})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.split("/") if p]

        if len(parts) == 3 and parts[0] == "api" and parts[1] == "feature-groups":
            name = parts[2]
            ok = self.store.delete_feature_group(name)
            if ok:
                self._send_json(200, {"message": f"Groupe '{name}' supprimé"})
            else:
                self._send_json(404, {"error": f"Groupe '{name}' introuvable"})
        else:
            self._send_json(404, {"error": "Route non trouvée"})


def start_api():
    """Démarre le serveur API en arrière-plan (thread daemon)."""
    global _server_thread, _httpd
    if _server_thread and _server_thread.is_alive():
        return  # déjà démarré

    _httpd = HTTPServer(("0.0.0.0", PORT), FeatureStoreHandler)
    _server_thread = threading.Thread(target=_httpd.serve_forever, daemon=True)
    _server_thread.start()


def stop_api():
    global _httpd
    if _httpd:
        _httpd.shutdown()


def get_api_url() -> str:
    return f"http://localhost:{PORT}"


if __name__ == "__main__":
    print(f"[API] Feature Store API démarrée sur http://localhost:{PORT}")
    start_api()
    import time
    while True:
        time.sleep(1)
