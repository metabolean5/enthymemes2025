# Enthymeme Detection from Tweets

This repository contains code and data for identifying enthymemes from tweets, focusing on implicit argumentation and rhetorical structure in socio-political discourse.

## Files and Structure

### Dataset
- `enthymemes2025gold.csv`  
  Main annotated dataset of tweets with enthymeme labels.

### RST-Augmented Features
- `dmrst_tropes_multifull.json`  
  RST features extracted using the DMRST parser.
- `dmrst_tropes_rstdt_gum_synthfull.json`  
  RST features based on the joint-joint-evaluation (JJE) structure typical of social media.  
  These two JSON files can be merged for model training.

### Feature Extraction
- `rst_features4.py`  
  Script for building RST features from the JSON files.

### Model Training & Evaluation
- Use the training scripts provided to train and evaluate different models.

### Error & Success Analysis
- `success_analysis3.py` — Analyze successful model predictions.
- `error_prediction3.py` — Analyze incorrect or failed predictions.

## Disclaimer
Some examples in the dataset may contain offensive content. The views expressed in these examples do not reflect those of the authors.
