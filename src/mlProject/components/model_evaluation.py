import os
import traceback
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import  save_json, initialize_mongodb, data_fetching, collection_deletion
from src.mlProject import logger

from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.config.model_path)

    def log_into_mlflow(self):
        try:
            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()
            
            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return
            
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            client, collection = initialize_mongodb("test")
            count = collection.count_documents({})
            metrix_dict = {}
            for i in range(count):
                test_data, name = data_fetching(collection,i)
                test_data.set_index(['Clock'], inplace=True, drop=True)
                    
                FEATURES = ['labeled_id', 'holiday', 'humidity', 'rain','cloud_cover', 'wind_speed', 'temp_diff',
                        'lag1', 'lag2', 'lag3','lag4', 'lag5', 'lag6', 'day', 'hour', 'month',
                        'dayofweek', 'quarter','dayofyear', 'weekofyear', 'year']
                TARGET = ['Kwh']
                test_x = test_data[FEATURES]
                test_y = test_data[TARGET]

                model = loaded_model_dict.get(i)

                if model is None:
                    logger.warning(f"Model for sensor {i} not found.")
                    continue

                with mlflow.start_run():
                    predicted_kwh = model.predict(test_x)
                    (rmse, mae, r2) = self.eval_metrics(test_y, predicted_kwh)

                    # Saving metrics as local
                    scores = {"rmse": rmse, "mae": mae, "r2": r2}
                    metrix_dict[f'{name}'] = scores

                    mlflow.log_params(self.config.all_params)

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("mae", mae)

                    # Model registry does not work with file store
                    if tracking_url_type_store != "file":
                        # Register the model
                        mlflow.sklearn.log_model(model, "model", registered_model_name=f"{i}_Model")
                    else:
                        mlflow.sklearn.log_model(model, "model")
            save_json(path=Path(self.config.metric_file_name), data = metrix_dict)
            collection_deletion(collection)

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())
        finally:
            client.close()
            logger.info("db connection closed")