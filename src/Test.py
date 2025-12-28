import mlflow
import dagshub

dagshub.init(
    repo_owner='Rocky0412',
    repo_name='Mlflow-experiment-track',
    mlflow=True
)

client = mlflow.tracking.MlflowClient()
print(client.list_artifacts("fda0f4793696420f92962ffacc09b6b5"))