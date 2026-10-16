# 🌊 AI Flood Resilience Network v2.0

> **Autonomous Multi-Agent Flood Early Warning, Deep Learning Risk Prediction & Emergency Coordination Platform**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg)](https://leafletjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

The **AI Flood Resilience Network** is an intelligent disaster preparedness and mitigation platform designed to bridge the gap between predictive hydrological analytics and real-time emergency field mobilization.

Rather than providing a simple binary flood prediction, this platform integrates **5 autonomous AI agents** managed by a central orchestrator. Together, they ingest meteorological, geographical, and topographical parameters, execute deep neural network inference, evaluate confidence bounds, formulate explainable risk factor attributions, broadcast real-time WebSocket alerts, and generate automated disaster response plans (NDRF deployment, high-capacity pump allocation, evacuation routes, and contextual safety guidelines).

---

## 🏗️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │    User / GIS Dashboard   │
                                  └─────────────┬─────────────┘
                                                │ REST / WebSockets
                                                ▼
                                  ┌───────────────────────────┐
                                  │     FastAPI Web Server    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │     CoordinationAgent     │
                                  │    (Master Orchestrator)  │
                                  └─────────────┬─────────────┘
                                                │
         ┌──────────────────────────────┬───────┴──────────────────────┬──────────────────────────────┐
         ▼                              ▼                              ▼                              ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ MonitoringAgent  │          │ PredictionAgent  │          │    AlertAgent    │          │  ResourceAgent   │
├──────────────────┤          ├──────────────────┤          ├──────────────────┤          ├──────────────────┤
│ • Micro-climate  │ ───────► │ • PyTorch MLP    │ ───────► │ • WebSocket Push │          │ • NDRF & Medical │
│ • Temp / Humidity│          │ • 20 Hydrological│          │ • Live Alert Log │          │ • Safe Routes    │
│ • Pressure & Wind│          │   Indicators     │          │ • Client Broadcast│         │ • Do's & Don'ts  │
│ • Real-time feeds│          │ • Confidence & XAI│         │                  │          │ • Helplines      │
└──────────────────┘          └──────────────────┘          └──────────────────┘          └──────────────────┘
```

---

## 🤖 The 5 Autonomous AI Agents

1. **`CoordinationAgent` (Master Orchestrator)**:
   - Manages the entire lifecycle: `Location Ingestion ➔ Weather Telemetry ➔ Neural Prediction ➔ Decision Gating ➔ Emergency Action Mobilization`.
   - Ensures decoupled agent communication and reports live system heartbeat status.

2. **`MonitoringAgent` (Meteorological Telemetry)**:
   - Evaluates micro-climatic conditions for any target coordinates.
   - Computes realistic atmospheric telemetry including temperature, "feels-like" index, humidity %, barometric pressure (hPa), wind velocity & vector direction, cloud cover, visibility (km), and UV index.

3. **`PredictionAgent` (Deep Learning & Explainable AI)**:
   - Executes a 20-feature PyTorch Multi-Layer Perceptron (`FloodMLP`) trained on global historical flood benchmarks.
   - Computes statistical confidence intervals (`score`, `upper_bound`, `lower_bound`, `margin_of_error`).
   - Conducts **Risk Factor Decomposition** to isolate the percentage contribution of Rainfall Intensity, Low Elevation, Slope Angle, and Water Proximity.

4. **`AlertAgent` (Real-Time Push Dispatcher)**:
   - Connects to frontend clients via WebSockets (`/ws/alerts/{client_id}`).
   - Automatically broadcasts instant warnings when predicted flood risk exceeds actionable thresholds (`MEDIUM` or `HIGH`).

5. **`ResourceAgent` (Emergency Logistics & Disaster Mitigation)**:
   - **Asset Allocation**: Mobilizes NDRF rescue units, Fire Brigade squads, mobile hospitals, 50,000 L/hr drainage pumps, and designated school/community shelters.
   - **Contextual Safety Guidelines**: Formulates risk-tailored Do's and Don'ts categorized by priority level (*Critical*, *High*, *Medium*).
   - **Evacuation Corridors**: Evaluates elevation contours to provide multi-directional evacuation paths (Direction, Distance, Route Open/Standby status).
   - **Emergency Helplines**: Integrates rapid-dial directories (NDRF, Police, Ambulance, Fire, National Disaster Helpline 1078).

---

## 🗺️ Interactive GIS & Dashboard Features

- **Multi-Basemap GIS Viewer**: Supports high-resolution **Satellite Imagery (Esri World Imagery)**, **Topographic Mapping (OpenStreetMap)**, and **Terrain Elevation Contours (OpenTopoMap)** without third-party API key watermarks.
- **Pan-India Hotspot Benchmarks**: Interactive markers for high-exposure zones (Mumbai, Chennai, Kolkata, Guwahati, Kochi, Delhi, Jodhpur, etc.) with pre-calibrated baseline topography.
- **Point-and-Click Geolocation**: Click anywhere on the map to automatically extract coordinates and populate the prediction engine.
- **Historical Trend Visualizer**: Built-in Chart.js analytics tracking probability distribution over consecutive runs.
- **Adaptive Dark / Light Themes**: High-contrast, accessibility-focused UI with native OS color-scheme styling.

---

## 📁 Repository Structure

```text
AI-Flood-Resilience-Network/
├── agents/
│   ├── alert_agent.py          # WebSocket broadcasting & notification dispatcher
│   ├── base_agent.py           # Base agent class with telemetry & status tracking
│   ├── coordination_agent.py   # Multi-agent orchestrator & pipeline workflow
│   ├── monitoring_agent.py     # Atmospheric simulation & weather telemetry
│   ├── prediction_agent.py     # PyTorch MLP inference & feature pipeline
│   └── resource_agent.py       # Emergency response planning, logistics & safety
├── data/
│   ├── raw/
│   │   └── global-flood-events-fao-eve.csv  # Historical FAO flood event aggregates
│   └── processed/
│       └── flood_training_data.csv          # Feature-engineered training dataset
├── models/
│   ├── flood_model_best.pth    # Serialized PyTorch neural network checkpoint
│   ├── prepare_data.py         # ETL, cleaning & feature engineering script
│   └── train_model.py          # Neural network training loop & validation
├── static/
│   └── index.html              # Interactive GIS dashboard & viewer (HTML/CSS/JS)
├── main.py                     # FastAPI application & entry point
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore definitions
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/riya-1408/AI-Flood-Resilience-Network.git
cd AI-Flood-Resilience-Network
```

### 3. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
python main.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the interactive GIS dashboard |
| `/predict` | `POST` | Triggers the complete 5-agent coordination & inference workflow |
| `/api/flood-zones` | `GET` | Retrieves known India flood hotspot benchmarks with baseline metrics |
| `/api/weather/{lat}/{lon}` | `GET` | Generates simulated weather telemetry for coordinate pairs |
| `/api/safety-tips/{risk_level}`| `GET` | Returns risk-tailored Do's, Don'ts, emergency helplines & evacuation routes |
| `/api/history` | `GET` | Retrieves historical prediction log for trend analytics |
| `/api/stats` | `GET` | Returns platform summary metrics (total runs, risk distributions, uptime) |
| `/agent-status` | `GET` | Health check reporting the live status of all 5 agents |
| `/ws/alerts/{client_id}` | `WebSocket` | Real-time bi-directional channel for instantaneous alert broadcasts |

---

## 🧪 Model Details & Training

The core inference engine uses a PyTorch Multi-Layer Perceptron (`FloodMLP`) featuring:
- Batch Normalization & Dropout (0.3) for regularization
- ReLU non-linearities and Sigmoid activation
- 20 input hydrological, topographic, and temporal indicators
- Trained against historical FAO flood event aggregates

To re-train or fine-tune the model:
```bash
# Prepare dataset
python models/prepare_data.py

# Train neural network
python models/train_model.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
