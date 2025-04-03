import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path):
    """Loads dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, text_column, label_column):
    """Prepares data by vectorizing text and splitting into training and testing sets."""
    X = df[text_column]
    y = df[label_column]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_transformed = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, vectorizer

def train_model(X_train, y_train):
    """Trains a Naïve Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)
    
    return accuracy, report

def main():
    file_path = "data_spam.csv"
    df = load_data(file_path)
    
    text_column = "message"
    label_column = "label"
    
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df, text_column, label_column)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()
