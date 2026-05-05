# Semantic Text Similarity Detection System

A machine learning project that detects semantic similarity between text pairs using TF-IDF and LightGBM.
## Overview
This project identifies whether two questions are semantically similar using classical machine learning techniques.

## Dataset
Quora Question Pairs (~400K samples)

## Tech Stack
- Python
- scikit-learn
- LightGBM
- NumPy / Pandas
- Matplotlib / Seaborn

## How to Use
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt

3. Load model and vectorizer:
   import pickle

   with open('final_quora_lgbm_model.pkl', 'rb') as f:
       model = pickle.load(f)

   with open('vectorizer.pkl', 'rb') as f:
       tfidf = pickle.load(f)

4. Use the prediction function (see notebook)

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
The confusion matrix shows that the model performs well in identifying non-duplicate questions and achieves balanced performance for duplicate detection.

- Accuracy: ~80%
- F1 Score: ~0.74 (duplicate class)
  
- <img width="709" height="574" alt="image" src="https://github.com/user-attachments/assets/74c37d39-6961-47da-a955-a1b7f4918006" />


## Key Highlights
- Hybrid feature engineering approach
- Scalable pipeline for large datasets
- Efficient handling of high-dimensional sparse data

## Future Improvements
- Use BERT / transformer models for better semantic understanding
