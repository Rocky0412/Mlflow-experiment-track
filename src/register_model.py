import mlflow
run_id='ad20482f393a4e3a92e9f6a3ccf18a7b'
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/DT_model",
    name="DecisionTreeClassifier"
)

