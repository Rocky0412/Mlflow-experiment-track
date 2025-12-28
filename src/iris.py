# Iris Dataset Machine Learning Project (End-to-End)

# 1. Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import dagshub

# Initialize Dagshub MLflow Integration
dagshub.init(
    repo_owner='Rocky0412',
    repo_name='Mlflow-experiment-track',
    mlflow=True
)

# Create / Choose Experiment
mlflow.set_experiment("Iris-Experiment")

# Start MLflow Run
with mlflow.start_run() as run:

    # 2. Load Dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    # Save dataset as CSV for logging
    df = X.copy()
    df["target"] = y
    df.to_csv("iris.csv", index=False)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Model Training
    model = DecisionTreeClassifier(criterion='gini', max_depth=5)
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Log Metrics & Params
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", 5)

    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="DecisionTreeModel"
    )

    # Log Artifacts
    mlflow.log_artifact("iris.csv", artifact_path="dataset")
    mlflow.log_artifact("src/iris.py")

    # Tags
    mlflow.set_tag("model_type", "DecisionTree")
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("author", "Rocky")

    # Save run ID
    run_id = run.info.run_id

# ------------------------------
# 6. Register the Model
# ------------------------------
print(f'run_id={run_id} ')
model_uri = f"runs:/{run_id}/DecisionTreeModel"
mlflow.register_model(model_uri, "IrisDecisionTree")

