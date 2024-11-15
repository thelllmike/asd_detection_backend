# Autism Detection Project

This project aims to detect traits of Autism Spectrum Disorder (ASD) based on various behavioral and demographic data. It uses machine learning to predict the likelihood of ASD traits based on questionnaire responses and demographic features.

## Features

- **Machine Learning Model**: Uses a trained `RandomForestClassifier` model to predict ASD traits.
- **FastAPI API**: A RESTful API built with FastAPI for serving predictions.
- **Preprocessing with Label Encoding**: Handles categorical data encoding for fields like `Sex`, `Ethnicity`, and `Family_mem_with_ASD`.
- **Scalability**: Built for easy deployment and integration with other systems.

## Prerequisites

- Python 3.7 or later
- [FastAPI](https://fastapi.tiangolo.com/)
- Joblib for model handling
- scikit-learn for preprocessing and machine learning

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/thelllmike/asd_detection_backend.git
cd autism-detection-project
