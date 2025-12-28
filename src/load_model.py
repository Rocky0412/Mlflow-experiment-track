import mlflow
import mlflow.pyfunc
import pandas as pd

df=pd.read_csv('iris.csv')
X_test = df.sample(1).drop(columns=['target']) 

# Set MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Correct URI
uri = "models:/Iris_DT/2"  # version 2

# Load the model
model = mlflow.pyfunc.load_model(uri)
print(f'model is {model}')
# Example: make predictions
predictions = model.predict(X_test)
print(predictions)

