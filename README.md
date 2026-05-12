<div align="center">

# Athlete Injury Risk Prediction
### Hybrid TCN + BiGRU + Transformer Deep Learning Framework for Sports Injury Forecasting

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Architecture-TCN%20%2B%20BiGRU%20%2B%20Transformer-6A0DAD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Sports%20AI-00897B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Injury%20Risk%20Prediction-D32F2F?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-2E7D32?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Dataset-Synthetic%20Triathlete%20Dataset-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Athletes-1000-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Time%20Series-Wearable%20Sensor%20Data-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Deployment-Ready-red?style=flat-square"/>
</p>

<br/>

> **A production-oriented hybrid deep learning framework for predicting athlete injury risk using multimodal wearable, physiological, and training-load time-series data. The system combines Temporal Convolutional Networks (TCN), Bidirectional GRUs, and Transformer self-attention to model both short-term workload spikes and long-range recovery dynamics in endurance athletes.**

</div>

---

# Table of Contents

- [Project Overview](#project-overview)
- [Why Injury Prediction Matters](#why-injury-prediction-matters)
- [Technical Highlights](#technical-highlights)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Training Configuration](#training-configuration)
- [Evaluation Framework](#evaluation-framework)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Inference Pipeline](#inference-pipeline)
- [Technology Stack](#technology-stack)
- [Engineering Principles](#engineering-principles)
- [Potential Extensions](#potential-extensions)
- [License](#license)
- [Author](#author)

---

# Project Overview

This project presents a complete deep learning pipeline for **athlete injury risk prediction** using longitudinal wearable sensor data, physiological biomarkers, recovery metrics, and training workload information collected from endurance athletes.

The framework is built around a custom hybrid neural architecture combining:

- **Temporal Convolutional Networks (TCN)** for local temporal workload pattern extraction
- **Bidirectional GRU layers** for sequential physiological dependency modeling
- **Transformer self-attention** for long-range temporal relationship learning

The system is designed to model the complex interaction between:

- Acute and chronic training load
- Recovery quality
- Sleep and heart-rate variability
- Fatigue accumulation
- Physiological stress adaptation
- Multi-session workload progression

Unlike traditional rule-based athlete monitoring systems, this framework learns nonlinear temporal injury signatures directly from athlete history.

The project includes:

- Full preprocessing pipeline
- Time-series feature engineering
- Hybrid deep learning architecture
- Threshold calibration
- Model serialization
- Evaluation dashboard
- Prediction export pipeline
- Deployment-ready artifacts

---

# Technical Highlights

## Architecture

- Hybrid TCN + BiGRU + Transformer architecture
- Dilated temporal convolutions for workload spike detection
- Bidirectional sequential physiological modeling
- Transformer attention for long-range temporal dependency learning
- Residual temporal feature propagation

## Data Engineering

- Multi-source athlete data fusion
- Sequential window generation
- Rolling workload metrics
- Recovery and fatigue trend engineering
- Leak-free normalization pipeline

## Training Robustness

- Threshold calibration
- Early stopping
- Checkpoint restoration
- Stratified evaluation
- Serialized preprocessing artifacts

## Deployment Readiness

- `best_model.pt`
- `best_threshold.pkl`
- `feature_cols.pkl`
- Prediction export pipeline
- Reproducible inference workflow

---

# Architecture Deep Dive

## Conceptual Pipeline

```text
Athlete Daily Metrics + Training Sessions
                |
                v
+------------------------------------------------+
|               PREPROCESSING                    |
|  - Merge athlete/activity/daily data           |
|  - Handle missing values                       |
|  - Temporal alignment                          |
|  - Feature engineering                         |
|  - Rolling workload metrics                    |
|  - Sequence generation                         |
+----------------------+-------------------------+
                       |
                       v
      Shape: (batch, sequence_len, features)

+------------------------------------------------+
|         STAGE 1 — TEMPORAL CONVOLUTION         |
|                                                |
|  Dilated Temporal Convolution Blocks           |
|  - Local workload spikes                       |
|  - Recovery fluctuations                       |
|  - High-frequency fatigue patterns             |
|  - Short-term temporal dependencies            |
+----------------------+-------------------------+
                       |
                       v

+------------------------------------------------+
|            STAGE 2 — BIDIRECTIONAL GRU         |
|                                                |
|  Sequential physiological modeling             |
|  - Recovery progression                        |
|  - HRV dynamics                                |
|  - Sleep adaptation trends                     |
|  - Fatigue accumulation                        |
+----------------------+-------------------------+
                       |
                       v

+------------------------------------------------+
|         STAGE 3 — TRANSFORMER ENCODER          |
|                                                |
|  Multi-head self-attention                     |
|  - Long-range temporal relationships           |
|  - Session interaction modeling                |
|  - Context-aware injury signatures             |
+----------------------+-------------------------+
                       |
                       v

+------------------------------------------------+
|             CLASSIFICATION HEAD                |
|                                                |
| Dense -> Dropout -> Sigmoid                    |
+----------------------+-------------------------+
                       |
                       v

            Predicted Injury Risk
```

---

# Dataset

## Synthetic Triathlete Dataset for Injury Prediction Research (2024)

This project uses the publicly available **Synthetic Triathlete Dataset for Injury Prediction Research (2024)** published on Zenodo by Leonardo Rossi.

The dataset contains synthetic but realistically generated longitudinal data for endurance athletes across the entire year of 2024.

## Dataset Components

| File | Description |
|---|---|
| `athletes.csv` | Athlete demographic and profile information |
| `daily_data.csv` | Daily physiological and biometric measurements |
| `activity_data.csv` | Timestamped training session records |

---

## Dataset Statistics

| Property | Value |
|---|---|
| Athletes | 1,000 |
| Daily Records | 366,000 |
| Activity Sessions | 384,153 |
| Time Span | Jan 1 2024 — Dec 31 2024 |
| Sports Domain | Endurance / Triathlon |
| Data Type | Synthetic wearable + training telemetry |
| License | CC BY 4.0 |

---

# Feature Engineering

## Temporal Features

- Rolling workload averages
- Acute workload
- Chronic workload
- Workload ratios
- Lag-based physiological trends
- Recovery trajectory features

## Recovery Features

- HRV trend analysis
- Sleep consistency
- Rest-day intervals
- Recovery imbalance metrics

## Fatigue Features

- Consecutive training stress
- High-intensity accumulation
- Training monotony indicators
- Session density metrics

---

# End-to-End Pipeline

```text
Step 1   Load athlete, daily, and activity datasets

Step 2   Merge multimodal athlete records

Step 3   Data cleaning and preprocessing
         |- Missing value handling
         |- Datetime conversion
         |- Temporal alignment

Step 4   Feature engineering
         |- Rolling workload metrics
         |- Recovery indicators
         |- Acute/chronic load ratios
         |- Sequential physiological trends

Step 5   Sequence generation
         |- Sliding temporal windows
         |- Athlete-wise sequence grouping

Step 6   Train / validation / test split

Step 7   Feature normalization
         |- Fit scaler on training data only
         +- Serialize scaler.pkl

Step 8   Hybrid model construction
         |- TCN blocks
         |- BiGRU encoder
         |- Transformer attention
         +- Dense classification head

Step 9   Model training
         |- Early stopping
         |- Threshold optimization
         |- Checkpoint saving

Step 10  Evaluation
          |- Accuracy
          |- ROC-AUC
          |- Precision/Recall/F1
          |- Confusion matrix

Step 11  Export
          |- best_model.pt
          |- scaler.pkl
          |- threshold.pkl
          +- prediction CSVs
```

---

# Training Configuration

| Parameter | Description |
|---|---|
| Framework | PyTorch |
| Architecture | TCN + BiGRU + Transformer |
| Task | Binary injury-risk prediction |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Threshold Calibration | Enabled |
| Early Stopping | Enabled |
| Serialized Model | `best_model.pt` |

---

# Repository Structure

```text
athlete-injury-risk-prediction/
|
+-- athlete_injury_risk_prediction_hybrid_dl.ipynb
|
+-- athletes.zip
+-- daily_data.zip
+-- activity_data.zip
|
+-- best_model.pt
+-- best_threshold.pkl
+-- scaler.pkl
+-- feature_cols.pkl
+-- history.pkl
+-- model_config.json
|
+-- test_predictions.csv
|
+-- README.md
+-- LICENSE
```

---

# Quickstart

## 1. Clone Repository

```bash
git clone https://github.com/sushantkothari/athlete-injury-risk-prediction.git

cd athlete-injury-risk-prediction
```

---

## 2. Install Dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn joblib tqdm
```

---

## 3. Run Notebook

Open:

```text
athlete_injury_risk_prediction_hybrid_dl.ipynb
```

Run all cells sequentially.

---

# Inference Pipeline

```python
import torch
import joblib
import numpy as np

# Load artifacts
model = torch.load("best_model.pt", map_location="cpu")

scaler = joblib.load("scaler.pkl")

threshold = joblib.load("best_threshold.pkl")

feature_cols = joblib.load("feature_cols.pkl")

# Example sequence input
sequence = np.random.rand(30, len(feature_cols))

# Normalize
sequence_scaled = scaler.transform(sequence)

# Convert to tensor
x = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

# Predict
model.eval()

with torch.no_grad():
    pred = model(x).squeeze().item()

risk = int(pred >= threshold)

print("Injury Risk Probability:", round(pred, 4))
print("Predicted Risk Label:", risk)
```

---

# Technology Stack

| Library | Role |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Evaluation and preprocessing |
| Matplotlib | Visualization |
| Seaborn | Statistical plotting |
| Joblib | Artifact serialization |

---

# Potential Extensions

- ONNX export
- TensorRT optimization
- FastAPI deployment
- Real-time wearable streaming inference
- Attention visualization
- SHAP feature attribution
- Athlete-specific injury interpretation

---

# License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

# Author

## Sushant Kothari

AI/ML Engineer • Data Scientist • Deep Learning Researcher

- GitHub: https://github.com/sushantkothari

---

<div align="center">

If this project helped you, consider starring the repository.

</div>
