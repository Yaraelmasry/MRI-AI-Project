# MRI-AI-Project Overview

An AI-powered clinical decision-support prototype for MRI brain tumor classification and preliminary reporting.

This project simulates the real workflow used in radiology departments:
PACS -> AI analysis -> radiologist validation -> RIS/EMR.

It includes:

A trained machine learning model

A full preprocessing + inference pipeline

An AI-generated preliminary report

A Streamlit web app

A radiologist review workflow (approve / reject / additional notes if needed)

Perfect for demonstrating applied AI, medical imaging understanding, and full-stack ML engineering.

# Features

AI Tumor Classification
Classifies MRI scans into: Glioma, Meningioma, and Pituitary tumor
Outputs prediction + confidence score
Model: RandomForestClassifier, trained on Kaggle MRI dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri


# Image Preprocessing Pipeline

Replicates training-time preprocessing: Convert to grayscale, Resize to 64×64, Flatten to feature vector, Normalize
Ensures inference = training pipeline


# AI-Generated Preliminary Report

After prediction, the system generates a text report containing:

Predicted tumor type

Confidence score

Timestamp

AI-disclaimer

Saved automatically to /reports/#

# Radiologist Review System

A mini RIS-like workflow:

Add radiologist notes
Approve or reject the AI report
If rejected — choose a reason (image quality, model misclassification, etc.)
Approved reports - /reports/validated/
Rejected reports - /reports/rejected/

This mimics how real radiologists supervise AI outputs.


# Live Streamlit Web App

The entire system is deployed using Streamlit Cloud.
Radiologists can:

Upload an MRI image

View AI results

Review and validate the report

Save decisions with notes
