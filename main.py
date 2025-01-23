import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import argparse
from colorama import Fore, Style

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400]  # 4400 items
test_data = data[4400:]    # 1172 items

# Train model
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Classifier.fit(vectorize_text, train_data.v1)

def predict_message(message):
    global Classifier
    global Vectorizer
    try:
        vectorize_message = Vectorizer.transform([message])
        predict = Classifier.predict(vectorize_message)[0]
        predict_proba = Classifier.predict_proba(vectorize_message).tolist()
        return predict, predict_proba
    except Exception as inst:
        return None, str(type(inst).__name__) + ' ' + str(inst)

def main():
    parser = argparse.ArgumentParser(description='Spam Message Predictor')
    parser.add_argument('msg', type=str, help='The message to classify')
    args = parser.parse_args()

    message = args.msg
    prediction, prediction_proba = predict_message(message)

    if prediction is not None:
        print("\nResults:")
        print("-------------------------------------")
        print(Fore.GREEN + f'Prediction: {prediction}' + Style.RESET_ALL)
        print(Fore.BLUE + f'Prediction Probabilities: {prediction_proba}' + Style.RESET_ALL)
    else:
        print(Fore.RED + f'Error: {prediction_proba}' + Style.RESET_ALL)

if __name__ == '__main__':
    main()