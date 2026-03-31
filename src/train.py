import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
import json
import argparse

STUDENT_NAME = "Your Name"
ROLL_NO = "2022BCS0005"

def get_data(version="v2", feature_set="all"):
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    if feature_set == "reduced":
        X = X[["sepal_length", "sepal_width"]]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train(run_name, dataset_version, model_type, n_estimators=100, max_depth=None, C=1.0, feature_set="all"):
    mlflow.set_experiment("2022BCS0005_experiment")
    with mlflow.start_run(run_name=run_name):
        X_train, X_test, y_train, y_test = get_data(dataset_version, feature_set)

        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = LogisticRegression(C=C, max_iter=200)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("feature_set", feature_set)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        metrics = {
            "name": STUDENT_NAME,
            "roll_no": ROLL_NO,
            "run_name": run_name,
            "accuracy": acc,
            "f1_score": f1
        }
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f)

        print(f"Run: {run_name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="run1")
    parser.add_argument("--dataset_version", default="v2")
    parser.add_argument("--model_type", default="random_forest")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--feature_set", default="all")
    args = parser.parse_args()
    train(args.run_name, args.dataset_version, args.model_type,
          args.n_estimators, args.max_depth, args.C, args.feature_set)