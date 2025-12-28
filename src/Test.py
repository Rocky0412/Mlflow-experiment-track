import mlflow
import dagshub

dagshub.init(
    repo_owner='Rocky0412',
    repo_name='Mlflow-experiment-track',
    mlflow=True
)



from mlflow.tracking import MlflowClient

client = MlflowClient()

run_id = "ad20482f393a4e3a92e9f6a3ccf18a7b"

for item in client.list_artifacts(run_id):
    print(item.path)