---
title: "End-to-End MLOps Case: Iris Predictor"
excerpt: "A machine learning engineering case study covering the full model lifecycle, from versioned training to deployment on orchestrated infrastructure."
header:
  teaser: # optional: add a screenshot/diagram path here
order: 2
---

## 🤖 Overview

**End-to-End MLOps Case: Iris Predictor** is a comprehensive Machine Learning Engineering case study demonstrating the full lifecycle of a model — from versioned training to deployment in an orchestrated infrastructure.

Repository: [github.com/LucasMVasconcelos/ml-engineer-case](https://github.com/LucasMVasconcelos/ml-engineer-case)

## 🎯 What was built

- **A training pipeline** that produces a versioned model artifact, tracked with DVC alongside the training data.
- **A FastAPI inference backend** serving predictions, with latency logging on every request.
- **A Streamlit dashboard** as an interactive frontend for end-users to try the model.
- **Data validation with Pydantic schemas**, so malformed requests are rejected before reaching the model.
- **A test suite** covering fairness, threshold, and parity testing — not just standard unit/integration tests, but checks aimed at the model's behavior itself.
- **Containerized local orchestration** with Docker and Docker Compose (API + UI as separate services).
- **Kubernetes manifests** for deploying the same services to a cluster.
- **A Jenkins CI/CD pipeline** automating tests and builds.
- **Observability**: every inference request is logged (timestamp, input features, predicted class, end-to-end latency) for later analysis.

## 🧰 Tech stack

Python · Scikit-Learn · Pandas · DVC · FastAPI · Streamlit · Docker & Docker Compose · Kubernetes · Jenkins · Pytest · Pydantic
