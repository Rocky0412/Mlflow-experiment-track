# Iris Dataset Machine Learning Project (End-to-End)

# 1. Import Libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os
import mlflow




# 2. Load Dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")




# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
mlflow.set_tracking_uri("http://localhost:5000")
# log some runs

mlflow.set_tracking_uri("file:./mlruns")
# log somewhere else

mlflow.set_experiment("Iris-Experiment")

with mlflow.start_run():
    
    model = DecisionTreeClassifier(criterion='gini',max_depth=5)
    model.fit(X_train, y_train)


    # 5. Model Evaluation
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric('accuracy',accuracy_score(y_test,y_pred))
    mlflow.log_param('max_depth',5)
    mlflow.sklearn.log_model(model,'Random forest')
    mlflow.log_artifact("src/iris.py")


    
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("dataset", "iris")
    mlflow.set_tag("author", "Rocky")




