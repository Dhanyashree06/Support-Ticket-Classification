"""
Support Ticket Classifier — Production Backend
REST API server with SQLite persistence, validation, and ML inference
"""

import http.server
import json
import sqlite3
import re
import string
import urllib.parse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DB_PATH   = BASE_DIR / "tickets.db"
STATIC_DIR = BASE_DIR / "static"

# ── ML Classifier (self-contained, no sklearn needed) ──
STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it","its",
    "they","them","their","what","which","who","this","that","these","those","am",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "a","an","the","and","but","if","or","as","of","at","by","for","with","about",
    "to","from","in","out","on","off","not","no","so","than","very","just","now",
    "hi","hello","dear","please","thank","thanks","team","support","help",
}

CATEGORY_KEYWORDS = {
    "Billing": [
        "charged","charge","payment","invoice","refund","billing","bill","subscription",
        "price","cost","fee","money","credit","debit","transaction","receipt","paid",
        "pay","overcharged","double","duplicate","amount","plan","upgrade","downgrade",
    ],
    "Technical Issue": [
        "bug","error","crash","broken","down","fail","not working","slow","issue",
        "problem","fix","loading","timeout","500","api","server","database","sync",
        "export","import","login","access","cannot","unable","stuck","freeze","lost",
        "missing","deleted","corrupted","production","deploy","integration","sdk",
    ],
    "Account": [
        "account","password","username","email","profile","login","signin","logout",
        "2fa","two factor","authentication","hacked","locked","security","permission",
        "role","admin","team","member","sso","oauth","delete","merge","transfer",
        "suspend","ban","verify","verification",
    ],
    "General Query": [
        "how","what","when","where","why","does","do","can","could","would","should",
        "difference","plan","feature","pricing","compare","policy","gdpr","compliance",
        "documentation","docs","tutorial","guide","demo","trial","discount","support",
        "hours","contact","faq","onboarding","integrate","compatible",
    ],
}

PRIORITY_SIGNALS = {
    "High": [
        "urgent","critical","emergency","immediately","asap","broken","down","hack",
        "hacked","breach","fraud","unauthorized","data loss","production","all users",
        "cannot access","completely","entire","everyone","corrupted","stolen",
    ],
    "Low": [
        "how do i","wondering","question","curious","when","where","what is","minor",
        "small","just","simple","quick","update profile","change name","fyi",
    ],
}

def clean_text(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\$[\d,.]+', 'money', text)
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]

def score_category(tokens: list[str]) -> tuple[str, float]:
    text = ' '.join(tokens)
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        scores[cat] = score
    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    confidence = round(scores[best] / total * 100, 1)
    return best, max(confidence, 35.0)

def score_priority(tokens: list[str], category: str) -> tuple[str, float]:
    text = ' '.join(tokens)
    for sig in PRIORITY_SIGNALS["High"]:
        if sig in text:
            return "High", 85.0
    for sig in PRIORITY_SIGNALS["Low"]:
        if sig in text:
            return "Low", 78.0
    # Category-based defaults
    defaults = {"Technical Issue": "Medium", "Billing": "Medium",
                "Account": "Medium", "General Query": "Low"}
    return defaults.get(category, "Medium"), 60.0

def predict(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Ticket text cannot be empty")
    if len(text.strip()) < 5:
        raise ValueError("Ticket text is too short (min 5 characters)")
    if len(text) > 2000:
        raise ValueError("Ticket text too long (max 2000 characters)")

    tokens = clean_text(text)
    category, cat_conf = score_category(tokens)
    priority, pri_conf = score_priority(tokens, category)

    action_map = {
        ("Technical Issue", "High"):  "🚨 Escalate to on-call engineer immediately",
        ("Billing",         "High"):  "💳 Route to billing team — process refund",
        ("Account",         "High"):  "🔐 Security alert — possible account compromise",
        ("General Query",   "High"):  "📋 Account manager follow-up within 1 hour",
        ("Technical Issue", "Medium"):"⚙️  Technical support queue (SLA: 4 hours)",
        ("Billing",         "Medium"):"💬 Billing specialist (SLA: 4 hours)",
        ("Account",         "Medium"):"👤 Account support (SLA: 4 hours)",
        ("General Query",   "Medium"):"📧 Standard support queue (SLA: 8 hours)",
        ("Technical Issue", "Low"):   "📝 Backlog — resolve within 3 days",
        ("Billing",         "Low"):   "📝 FAQ link + 3-day response",
        ("Account",         "Low"):   "📝 Documentation link + 3-day response",
        ("General Query",   "Low"):   "📝 Auto-reply with FAQ",
    }

    return {
        "category":             category,
        "category_confidence":  cat_conf,
        "priority":             priority,
        "priority_confidence":  pri_conf,
        "recommended_action":   action_map.get((category, priority), "📬 General support queue"),
    }


# ── Database ────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tickets (
                id          TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                category    TEXT NOT NULL,
                priority    TEXT NOT NULL,
                cat_conf    REAL NOT NULL,
                pri_conf    REAL NOT NULL,
                action      TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                status      TEXT DEFAULT 'open'
            );
            CREATE TABLE IF NOT EXISTS stats (
                date        TEXT PRIMARY KEY,
                total       INTEGER DEFAULT 0,
                high        INTEGER DEFAULT 0,
                medium      INTEGER DEFAULT 0,
                low         INTEGER DEFAULT 0
            );
        """)
    print(f"  ✅ Database ready: {DB_PATH}")

def save_ticket(ticket_id: str, text: str, result: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    today = now[:10]
    with get_db() as conn:
        conn.execute("""
            INSERT INTO tickets (id, text, category, priority, cat_conf, pri_conf, action, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticket_id, text, result["category"], result["priority"],
              result["category_confidence"], result["priority_confidence"],
              result["recommended_action"], now))
        conn.execute("""
            INSERT INTO stats (date, total, high, medium, low) VALUES (?, 1, 0, 0, 0)
            ON CONFLICT(date) DO UPDATE SET
                total  = total + 1,
                high   = high   + CASE WHEN ? = 'High'   THEN 1 ELSE 0 END,
                medium = medium + CASE WHEN ? = 'Medium' THEN 1 ELSE 0 END,
                low    = low    + CASE WHEN ? = 'Low'    THEN 1 ELSE 0 END
        """, (today, result["priority"], result["priority"], result["priority"]))
    return {"id": ticket_id, "created_at": now, **result}

def get_tickets(limit=50, offset=0, priority=None, category=None) -> list:
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []
    if priority:
        query += " AND priority = ?"; params.append(priority)
    if category:
        query += " AND category = ?"; params.append(category)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params += [limit, offset]
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]

def get_ticket(ticket_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    return dict(row) if row else None

def get_summary() -> dict:
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        by_cat = conn.execute(
            "SELECT category, COUNT(*) as count FROM tickets GROUP BY category").fetchall()
        by_pri = conn.execute(
            "SELECT priority, COUNT(*) as count FROM tickets GROUP BY priority").fetchall()
        recent = conn.execute(
            "SELECT * FROM tickets ORDER BY created_at DESC LIMIT 5").fetchall()
    return {
        "total_tickets": total,
        "by_category":   {r["category"]: r["count"] for r in by_cat},
        "by_priority":   {r["priority"]: r["count"] for r in by_pri},
        "recent":        [dict(r) for r in recent],
    }


# ── HTTP Helpers ────────────────────────────────────────
def json_response(handler, status: int, data: dict):
    body = json.dumps(data, indent=2).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)

def error(handler, status: int, message: str):
    json_response(handler, status, {"error": message, "status": status})

def parse_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length)
    return json.loads(raw) if raw else {}

def parse_query(path: str) -> tuple[str, dict]:
    parsed = urllib.parse.urlparse(path)
    params = dict(urllib.parse.parse_qsl(parsed.query))
    return parsed.path, params


# ── Request Handler ─────────────────────────────────────
class APIHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path, params = parse_query(self.path)

        # Serve frontend
        if path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if not index_path.exists():
                index_path = BASE_DIR / "index.html"
            self._serve_file(index_path, "text/html")
            return

        # ── GET /api/health
        if path == "/api/health":
            json_response(self, 200, {
                "status": "ok",
                "version": "2.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # ── GET /api/tickets
        elif path == "/api/tickets":
            try:
                limit    = min(int(params.get("limit", 50)), 200)
                offset   = int(params.get("offset", 0))
                priority = params.get("priority")
                category = params.get("category")
                tickets  = get_tickets(limit, offset, priority, category)
                json_response(self, 200, {"tickets": tickets, "count": len(tickets)})
            except Exception as e:
                error(self, 400, str(e))

        # ── GET /api/tickets/{id}
        elif re.match(r"^/api/tickets/[\w-]+$", path):
            ticket_id = path.split("/")[-1]
            ticket = get_ticket(ticket_id)
            if ticket:
                json_response(self, 200, ticket)
            else:
                error(self, 404, f"Ticket '{ticket_id}' not found")

        # ── GET /api/summary
        elif path == "/api/summary":
            json_response(self, 200, get_summary())

        else:
            error(self, 404, f"Route '{path}' not found")

    def do_POST(self):
        path, _ = parse_query(self.path)

        # ── POST /api/classify
        if path == "/api/classify":
            try:
                body = parse_body(self)
                text = body.get("text", "").strip()

                # Validate
                if not text:
                    return error(self, 422, "Field 'text' is required")
                if len(text) < 5:
                    return error(self, 422, "Ticket text too short (min 5 characters)")
                if len(text) > 2000:
                    return error(self, 422, "Ticket text too long (max 2000 characters)")

                # Predict
                result = predict(text)

                # Persist
                ticket_id = str(uuid.uuid4())[:8].upper()
                saved = save_ticket(ticket_id, text, result)

                json_response(self, 200, {
                    "success": True,
                    "ticket": saved,
                })

            except json.JSONDecodeError:
                error(self, 400, "Invalid JSON body")
            except ValueError as e:
                error(self, 422, str(e))
            except Exception as e:
                error(self, 500, f"Internal server error: {e}")

        else:
            error(self, 404, f"Route '{path}' not found")

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            error(self, 404, "File not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ── Entry point ─────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8000))
    STATIC_DIR.mkdir(exist_ok=True)
    print(f"  🔍 Serving static files from: {STATIC_DIR.resolve()}")
    init_db()

    print(f"""
╔══════════════════════════════════════════════╗
║   Support Ticket Classifier — Backend v2.0   ║
╠══════════════════════════════════════════════╣
║   http://localhost:{PORT:<4}                       ║
║                                              ║
║   Endpoints:                                 ║
║   GET  /api/health          health check     ║
║   POST /api/classify        classify ticket  ║
║   GET  /api/tickets         list tickets     ║
║   GET  /api/tickets/:id     get ticket       ║
║   GET  /api/summary         stats + recent   ║
╚══════════════════════════════════════════════╝
""")
    try:
        server = http.server.HTTPServer(("", PORT), APIHandler)
        server.serve_forever()
    except OSError as e:
        if e.errno == 98 or e.errno == 10048:
            print(f"  ❌ ERROR: Port {PORT} is already in use. Please stop other processes or use a different port.")
        else:
            print(f"  ❌ ERROR starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Server stopped.")
