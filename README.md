# Semantic Text Similarity Detection System

## Overview
This project identifies whether two questions are semantically similar using classical machine learning techniques.

## Dataset
Quora Question Pairs (~400K samples)

## Features
- TF-IDF vectorization
- Cosine similarity
- Word overlap
- Length-based features

## Models Used
- Logistic Regression
- SVM
- XGBoost
- Ensembled with Hard Voting
- LightGBM (final model)

## Results
- Accuracy: ~80%
- F1 Score: ~0.74 (duplicate class)
- <img width="709" height="574" alt="image" src="https://github.com/user-attachments/assets/74c37d39-6961-47da-a955-a1b7f4918006" />


## Key Highlights
- Hybrid feature engineering approach
- Scalable pipeline for large datasets
- Efficient handling of high-dimensional sparse data

## Future Improvements
- Use BERT / transformer models for better semantic understanding
