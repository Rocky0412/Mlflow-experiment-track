from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
import os

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

mlflow.set_tracking_uri(uri='http://127.0.0.1:5000')

experiment_name = "Iris_Experiment"
mlflow.set_experiment(experiment_name)



# Start MLflow run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # --------------------------
    # Load Dataset
    # --------------------------
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    # Save dataset as CSV
    df = X.copy()
    df["target"] = y
    dataset_path = "iris.csv"
    df.to_csv(dataset_path, index=False)

    # --------------------------
    # Train-Test Split
    # --------------------------
    test_size = 0.3
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # --------------------------
    # Model Training
    # --------------------------
    max_depth = 5
    criterion = "gini"
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)

    # --------------------------
    # Model Evaluation
    # --------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --------------------------
    # Log Parameters, Metrics, Tags
    # --------------------------
    mlflow.log_params({
        "criterion": criterion,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state,
        "num_rows": df.shape[0],
        "num_features": df.shape[1]
    })
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tags({
        "model_type": "DecisionTree",
        "dataset": "iris",
        "author": "Rocky"
    })

    # --------------------------
    # Log Model
    # --------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="Iris_DT"
    )

    # --------------------------
    # Log Artifacts
    # --------------------------
    mlflow.log_artifact(dataset_path, artifact_path="dataset")
    script_path = os.path.abspath(__file__) if os.path.exists(__file__) else None
    if script_path:
        mlflow.log_artifact(script_path, artifact_path="src")

    # --------------------------
    # Verify Artifacts
    # --------------------------
    


#---- Model Register ------------------------
model_uri = f"runs:/{run_id}/Iris_DT"
registered_model_name = "Iris_DT"

result = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name,
    
)

print(f'model is register with the version {result.version}')