"""
Machine Learning Model - Classification Example
This module demonstrates a simple machine learning classification model
using scikit-learn's Iris dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """Load and return the Iris dataset."""
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    return accuracy, report


def main():
    """Main function to run the ML pipeline."""
    # Load data
    X, y, target_names = load_data()
    print("Data loaded successfully!")
    print(f"Dataset shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model
    model = train_model(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate model
    accuracy, report = evaluate_model(model, X_test, y_test, target_names)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(report)

    return model


if __name__ == "__main__":
    main()
