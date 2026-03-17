# 🎫 Ticket Triage — Backend v2.0

Production-grade Python backend for the Support Ticket Classifier.
Zero external dependencies — runs on Python 3.10+ stdlib only.

## 🚀 Quick Start

```bash
python3 server.py
# → http://localhost:8000
```

## 📡 REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | /api/health | Server health check |
| POST | /api/classify | Classify ticket (saves to DB) |
| GET  | /api/tickets | List tickets (filterable) |
| GET  | /api/tickets/:id | Get single ticket |
| GET  | /api/summary | Stats + recent tickets |

### POST /api/classify
Request: `{ "text": "I was charged twice this month." }`

Response:
```json
{
  "success": true,
  "ticket": {
    "id": "A1B2C3D4",
    "category": "Billing",
    "category_confidence": 80.0,
    "priority": "High",
    "priority_confidence": 85.0,
    "recommended_action": "Route to billing team",
    "created_at": "2025-03-12T10:30:00+00:00"
  }
}
```

### GET /api/tickets
Query params: `?priority=High&category=Billing&limit=50&offset=0`

## 🗄️ Database
SQLite — auto-created as `tickets.db` on first run.

## 🏗️ Structure
```
backend/
├── server.py          REST API + ML classifier + DB logic
├── static/
│   └── index.html     Dark industrial frontend UI
└── tickets.db         Auto-created SQLite database
```
