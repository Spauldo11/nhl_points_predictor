---
title: NHL Points Predictor
emoji: 🏒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# NHL Points Predictor Web Application

A machine learning web application that predicts future NHL player point totals using Keras neural network models trained on historical player statistics.

## Deployment on Hugging Face Spaces

This repository is configured for Hugging Face Spaces Docker SDK listening on port `7860`.

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the web server:
   ```bash
   python server.py
   ```
3. Open `http://localhost:5000` in your browser.
