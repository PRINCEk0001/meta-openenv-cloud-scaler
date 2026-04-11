---
title: Cloud AutoScaler & Security Auditor
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ☁️ Anigrevity: Multi-Agent OpenEnv Suite

[![Gymnasium Compliant](https://img.shields.io/badge/Gymnasium-v0.29.1-blue.svg)](https://gymnasium.farama.org/)
[![FastAPI Server](https://img.shields.io/badge/FastAPI-0.110.0-teal.svg)](https://fastapi.tiangolo.com)
[![Meta OpenEnv](https://img.shields.io/badge/Track-Meta_OpenEnv-purple.svg)]()

Welcome to the **Anigrevity OpenEnv Suite**, a collection of three high-fidelity simulation environments designed for the **Meta OpenEnv Track** hackathon. 

This repository implements a production-ready, agent-agnostic infrastructure for evaluating LLM-based autonomous controllers across infrastructure management, security auditing, and failure diagnosis.

---

## 🎭 The Anigrevity Persona

All agents in this suite are integrated with the **Anigrevity** persona—a strict, goal-oriented autonomous controller designed for Zero-Shot and Few-Shot LLM evaluation.

1.  **Cloud Infrastructure Controller (Scaler)**: Manages server fleets to balance 70% utilization with <50ms latency.
2.  **Expert Security Auditor (Reviewer)**: Scans 5-step diffs for SQLi, XSS, and auth bypasses. Blocks vulnerabilities with `reject` actions.
3.  **ML Failure Diagnosis (WDIF)**: Diagnoses training failures (e.g., vanishing gradients, dying ReLUs) through systematic log and config inspection.

---

## 🏗️ Technical Architecture & Requirements

### 💎 Strict Compliance (Phase 2 Hardened)
This repository is optimized to pass the Meta OpenEnv automated validator with a **100% success rate** on boundary conditions:
*   **Hardened Reward Range**: All rewards and final task scores are strictly clamped to the **(0.01, 0.99)** interval. Values like `0.0` or `1.0` are impossible, preventing evaluation "out of range" errors.
*   **Standardized Logging**: Every episode produces a mandatory `[END]` log in the exact format required: `[END] task={task} score={final_score:.2f} steps={n}`.
*   **LLM Proxy Integration**: Ready for the `HF_TOKEN` and `API_BASE_URL` injection, using the Groq-powered `llama-3.1-8b-instant` baseline.

---

## 📊 Environment Catalog

### 1. Cloud AutoScaler (`autoscaling_easy/medium/hard`)
Simulates a core dev-ops challenge: managing server farms in response to volatile web traffic.
*   **Actions**: `0: Hold`, `1: Scale Up`, `2: Scale Down`.
*   **Optimization**: Target 60-80% capacity utilization.
*   **Episode Horizon**: 50 steps.

### 2. Code Review Auditor (`code_review_easy/medium/hard`)
Evaluates security reasoning by presenting patches with potential vulnerabilities.
*   **Actions**: `approve`, `reject`, `request_changes`, `comment`.
*   **Episode Horizon**: 5 steps.

### 3. Why Did It Fail (`whydiditfail_easy`)
ML Diagnosis task requiring step-by-step investigation of training logs and configurations.
*   **Actions**: `inspect_logs`, `inspect_config`, `inspect_gradients`, `submit_diagnosis`.

### 4. Kinetic Console Dashboard (`index.html`)
A high-fidelity, pixel-perfect frontend for real-time monitoring.
*   **Features**: Line charts for rewards, scrolling terminal logs, and clickable action controls.
*   **Access**: Open `index.html` in any browser while the backend is running.

---

## 🚀 Deployment & Usage

### 🎨 Kinetic Dashboard
To view the real-time telemetry console:
1.  Ensure the FastAPI server is running (`python -m server.app`).
2.  Open **[index.html](file:///e:/Meta%20Ai/index.html)** in your browser.
3.  The console will automatically sync with the backend at `localhost:7860`.

---

## 🚀 Deployment & Usage

### HF Spaces Standard
The environment serves a FastAPI wrapper on port `7860`.
```bash
docker build -t meta-openenv-anigrevity .
docker run -p 7860:7860 meta-openenv-anigrevity
```

### Integrated Inference
To run a full evaluation:
```bash
set HF_TOKEN=your_token_here
python inference.py
```

### Pre-Deployment Verification
```bash
python main.py
```

---
*Developed for the Meta LLM OpenEnv Hackathon* ☁️
Meta LLM OpenEnv Hackathon* ☁️
