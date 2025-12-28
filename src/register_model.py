import mlflow
path='models/m-e3cc9317d5024c65be38aad01111263a/artifacts/model.pkl'
run_id='e40bbcf3c7184e21b51ad3c144d8954d'
model_uri=f'runs:/{run_id}/model'
mlflow.register_model(model_uri=model_uri, name="DecisionTree_final")
