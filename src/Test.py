import mlflow
import dagshub

dagshub.init(
    repo_owner='Rocky0412',
    repo_name='Mlflow-experiment-track',
    mlflow=True
)



from mlflow.tracking import MlflowClient

client = MlflowClient()
print(client.list_artifacts("8aabd3e394d846f49ead8a536ac75322"))