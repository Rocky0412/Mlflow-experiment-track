import mlflow
import dagshub

dagshub.init(
    repo_owner='Rocky0412',
    repo_name='Mlflow-experiment-track',
    mlflow=True
)



from mlflow.tracking import MlflowClient

client = MlflowClient()

run_id = "e40bbcf3c7184e21b51ad3c144d8954d"

for item in client.list_artifacts(run_id):
    print(item.path)