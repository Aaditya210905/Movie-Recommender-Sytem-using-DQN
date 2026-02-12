# 🎬 DQN Movie Recommendation System

A **Deep Q-Network (DQN)** based movie recommendation engine trained on **7,841 IMDB movies**, featuring an interactive Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)

---

## Overview

This project applies **Reinforcement Learning** to the movie recommendation problem. A Dueling DQN agent learns to map user preference states to movie selections, optimising for simulated user feedback signals (like, watch, click, skip, ignore).

### Key Components

| Component | Details |
|---|---|
| **State** | 310-D user preference vector (209 genre dims + 100 TF-IDF/SVD embeddings + 1 rating) |
| **Action** | Select one of 7,841 movies |
| **Reward** | like (+1.5) · watch (+0.8) · click (+0.4) · skip (−0.2) · ignore (−0.5) |
| **Architecture** | Dueling DQN with LeakyReLU, BatchNorm, Dropout |
| **Training** | ε-greedy exploration, experience replay, target network sync |

---

## Project Structure

```
├── app.py                          # Streamlit web interface
├── core.py                         # DQN agent, environment & recommender classes
├── main.ipynb                      # Training notebook (data processing → training → evaluation)
├── imdb_movies_2025_cleaned.csv    # Cleaned IMDB dataset (7,841 movies)
├── models/                         # Trained model weights & serialised objects
│   ├── dqn_recommender_q_network.keras
│   ├── dqn_recommender_target_network.keras
│   ├── dqn_recommender_params.pkl
│   ├── feature_processor.pkl
│   ├── movie_catalog.pkl
│   └── training_history.pkl
├── agent_comparison.png            # Agent comparison visualisation
├── network_architecture_fixed.png  # Network architecture diagram
├── dqn_network_visualization.html  # Interactive network visualisation
├── network_activation_flow.html    # Activation flow visualisation
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/dqn-movie-recommender.git
cd dqn-movie-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

---

## App Pages

| Page | Description |
|---|---|
| 🏠 **Home** | System overview and quick stats |
| 🔬 **Pipeline Visualizer** | End-to-end RL pipeline walkthrough |
| 🎯 **Get Recommendations** | Cold-start top-K from the DQN |
| 🎭 **Genre Explorer** | Genre-biased personalised picks |
| ⚡ **Hybrid Recommender** | Blend Q-values with content similarity |
| 💬 **Interactive Session** | Give feedback and watch the model adapt in real time |
| 📊 **Training Dashboard** | Loss curves, reward history, epsilon decay |
| 🧠 **Model Architecture** | Layer-by-layer network details |
| 📈 **Dataset Analytics** | Explore the IMDB movie dataset |

---

## Training

The full training pipeline is in `main.ipynb`. It covers:

1. **Data loading & cleaning** — IMDB 2025 dataset
2. **Feature engineering** — genre binarisation, TF-IDF + SVD, rating scaling
3. **Environment setup** — simulated user feedback loop
4. **DQN training** — Dueling DQN with experience replay
5. **Evaluation** — reward curves, recommendation quality analysis

---

## Tech Stack

- **TensorFlow / Keras** — DQN model (Dueling architecture)
- **Streamlit** — Interactive web UI
- **scikit-learn** — Feature processing (TF-IDF, SVD, scaling)
- **Plotly** — Visualisations
- **NumPy / Pandas** — Data handling

---

## License

This project is for educational purposes (Semester 4 — Reinforcement Learning).
