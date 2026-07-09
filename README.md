# 🤟 SignAI — AI-Powered Sign Language Communication Platform

> Real-time American Sign Language (ASL) recognition from webcam video, converting gestures to text and speech for accessibility.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00897B)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Project Overview

SignAI bridges the communication gap for the hearing-impaired community by providing:

- 🎯 **Real-time gesture recognition** — 30fps webcam pipeline using MediaPipe + LSTM
- 🗣️ **Text-to-Speech** — recognized signs converted to natural speech in 5 languages
- 📊 **Analytics dashboard** — session accuracy, usage trends, sign frequency charts
- 🔐 **User accounts** — JWT authentication, conversation history, personal settings
- 🌐 **Multilingual** — English, Tamil, Hindi, Spanish, French translation

**Recognition accuracy:** >90% on 50+ ASL signs  
**End-to-end latency:** <200ms  
**Tech stack:** Python · OpenCV · MediaPipe · TensorFlow · FastAPI · PostgreSQL · React · Tailwind

---

## 🏗️ Architecture

```
[ React + MediaPipe (browser) ]
          ↕ WebSocket
[ FastAPI Backend ]  ←→  [ PostgreSQL + Redis ]
          ↕
[ LSTM / Transformer Model ]
          ↕
[ TTS + Google Translate ]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 15 (Week 4+)
- Webcam

### Backend Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/sign-language-platform.git
cd sign-language-platform

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt

# Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your database credentials

# Run backend
cd backend
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
# App: http://localhost:5173
```

### CV Demo (Week 2 — no backend needed)
```bash
# From project root, with venv active
python ml_training/data_collection/run_demo.py
```

---

## 📁 Project Structure

```
sign-language-platform/
├── backend/               # FastAPI REST + WebSocket API
│   ├── app/
│   │   ├── main.py        # App entry point
│   │   ├── config.py      # Settings via pydantic-settings
│   │   ├── models/        # SQLAlchemy ORM models
│   │   ├── routers/       # API route handlers
│   │   ├── services/      # Business logic
│   │   └── ml/            # Model loading + inference
│   └── tests/             # Pytest test suite
├── frontend/              # React + Tailwind CSS
│   └── src/
│       ├── pages/         # Route pages
│       ├── components/    # UI components
│       ├── hooks/         # Custom React hooks
│       └── store/         # Zustand state
├── ml_training/           # Model training pipeline
│   ├── data_collection/   # Landmark extraction scripts
│   ├── models/            # LSTM / Transformer definitions
│   └── saved_models/      # Trained .h5 / .pt files
└── dataset/               # Preprocessed landmark data
```

---

## 🗓️ 16-Week Roadmap

| Phase | Weeks | Focus |
|-------|-------|-------|
| Foundation | 1–4 | Env setup, MediaPipe, database, auth |
| ML Model | 5–8 | LSTM training, inference API, WebSocket |
| Full Stack | 9–12 | React integration, TTS, dashboard |
| Deploy | 13–16 | Advanced features, Docker, live deploy |

---

## 🧠 ML Pipeline

**Input:** 21 hand landmarks (x, y, z) × 30 frames = 630 features per sequence  
**Model:** Bidirectional LSTM (128 → 64 units) with dropout  
**Output:** Softmax over N sign classes  

**Datasets used:**
- [WLASL](https://github.com/dxli94/WLASL) — 2000 ASL words from video
- [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — 29 static signs
- Custom self-recorded gestures

---

## 🔌 API Reference

Full interactive docs available at `http://localhost:8000/docs` when the backend is running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/auth/login` | POST | JWT login |
| `/api/auth/register` | POST | Create account |
| `/ws/stream/{id}` | WebSocket | Real-time landmark stream |
| `/api/history/sessions` | GET | Conversation history |
| `/api/analytics/summary` | GET | Dashboard stats |

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

---

## 🐳 Docker

```bash
# Full stack (backend + frontend + postgres + redis)
docker-compose up --build
```

---

## 📄 License

MIT © 2024 — Built as a final year computer science project.

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev) — Google's hand landmark model
- [WLASL Dataset](https://github.com/dxli94/WLASL) — Word-Level American Sign Language
- [FastAPI](https://fastapi.tiangolo.com) — Modern Python web framework
