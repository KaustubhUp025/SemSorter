# SemSorter — AI Hazard Sorting System

> **Real-time robotic arm simulation controlled by a multimodal AI agent using the [Vision-Agents SDK](https://github.com/GetStream/vision-agents) by GetStream.**

[![Demo](https://img.shields.io/badge/Live%20Demo-Render.com-4f46e5)](https://semsorter.onrender.com)
[![GitHub](https://img.shields.io/badge/GitHub-KaustubhUp025-181717?logo=github)](https://github.com/KaustubhUp025/SemSorter)

---

## 🤖 Overview

SemSorter is an AI-powered hazardous waste sorting system where a Franka Panda robotic arm, simulated in MuJoCo, is controlled by a multimodal AI agent. The agent:

1. **Watches** the conveyor belt via a live camera feed
2. **Detects** hazardous items (flammable / chemical) using **Gemini VLM**
3. **Plans and executes** pick-and-place operations via **Gemini LLM function-calling**
4. **Speaks back** results using **ElevenLabs TTS**
5. **Listens** to voice commands via **Deepgram STT**

All orchestration uses the **[Vision-Agents SDK](https://github.com/GetStream/vision-agents)** by GetStream.

---

## 🏗 Architecture

```
Browser  ←─── WebSocket ───→  FastAPI Server
                                    │
                          Vision-Agents SDK Agent
                          ┌─────────┴──────────┐
                     gemini.LLM          deepgram.STT
                     (tool-calling)      (voice→text)
                          │
                     VLM Bridge
                          │
                     MuJoCo Sim (Franka Panda)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MuJoCo 3.x
- EGL (headless GPU rendering)

### Local Setup

```bash
# Clone
git clone https://github.com/KaustubhUp025/SemSorter.git
cd SemSorter

# Install dependencies
pip install -r requirements-server.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
# GOOGLE_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY
# STREAM_API_KEY, STREAM_API_SECRET

# Run
MUJOCO_GL=egl uvicorn SemSorter.server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

### Voice Agent (Vision-Agents SDK CLI)
```bash
cd Vision-Agents
MUJOCO_GL=egl uv run python ../SemSorter/agent/agent.py run
```

---

## 📦 Project Structure

```
SemSorter/
├── SemSorter/
│   ├── simulation/
│   │   ├── controller.py          # MuJoCo sim + IK + pick-and-place
│   │   └── semsorter_scene.xml    # MJCF scene (Panda + conveyor + bins)
│   ├── vision/
│   │   ├── vision_pipeline.py     # Gemini VLM hazard detection
│   │   └── vlm_bridge.py         # VLM → sim item matching
│   ├── agent/
│   │   ├── agent.py               # Vision-Agents SDK agent
│   │   └── semsorter_instructions.md
│   └── server/
│       ├── app.py                 # FastAPI + WebSocket video stream
│       ├── agent_bridge.py        # SDK bridge + quota detection
│       └── static/index.html      # Web UI
├── Vision-Agents/                 # GetStream Vision-Agents SDK
├── Dockerfile
├── render.yaml
└── requirements-server.txt
```

---

## 🔑 API Keys Required

| Service | Purpose | Free tier |
|---|---|---|
| Google Gemini | LLM orchestration + VLM detection | 15 RPM |
| Deepgram | Speech-to-Text | 45 min/month |
| ElevenLabs | Text-to-Speech | ~10k chars/month |
| GetStream | Real-time video call (Voice agent) | Free tier available |

> **API exhaustion handling:** The server detects quota errors (`429 / ResourceExhausted`) and automatically switches to demo-mode per service, showing a banner in the UI.

---

## 🐳 Deploy to Render

1. Fork this repo
2. Create a new **Web Service** on [Render.com](https://render.com) pointing to your fork
3. Add your API keys as **Environment Variables** in the Render dashboard
4. Done — Render auto-deploys from `render.yaml`
