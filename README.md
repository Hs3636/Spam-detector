# Spam Prediction SMS Classifier

## Overview
This project implements a machine learning-based SMS spam classifier using Support Vector Classification (SVC) and TF-IDF text representation.

## Prerequisites
- Python 3.7+
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hs3636/Spam-detector.git
cd Spam-detector
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

### Classify a Single Message
```bash
python app.py "This is a message to classify"
```
Output will include the predicted class (spam/ham) and probability.

### Test Model Accuracy
```bash
python test_accuracy.py
```
Generates `test_accuracy.csv` with prediction results.

## Model Details
- Algorithm: Support Vector Classification (SVC)
- Feature Representation: TF-IDF
- Test Dataset Accuracy: ~98%

## Potential Improvements
- Explore advanced feature engineering
- Experiment with alternative ML algorithms
- Implement cross-validation
- Enhance error handling

## Dataset
Trained on Spam/Ham SMS Dataset from Kaggle.

Link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

