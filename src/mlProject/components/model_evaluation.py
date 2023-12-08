import os
import traceback
import datetime
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import create_features, add_lags, store_actual_data, store_predictions_in_mongodb
from src.mlProject import logger


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

    def actualData(self, data_sensor):
        ''' Dumping Previous month Transformed data into mongo db for Actual vs Predited graph'''
        end_date = datetime.datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_date = (end_date - datetime.timedelta(days=end_date.day)).replace(day=1, hour=0, minute=0,
                                                                                second=0, microsecond=0)

        data_sensor['Clock'] = pd.to_datetime(data_sensor['Clock'])
        last_month_data = data_sensor[(data_sensor['Clock'] >= start_date) & (data_sensor['Clock'] < end_date)]
        store_actual_data(last_month_data)
        return

    def log_into_mlflow(self):
        try:
            data_files = [file for file in os.listdir(self.config.test_data_path) if file.startswith('sensor')]
            data_list = []
            for data_file in data_files:
                data_sensor = pd.read_csv(os.path.join(self.config.test_data_path, data_file))
                # self.actualData(data_sensor)
                data_list.append(data_sensor)

            # Concatenate data for all sensors
            train_data = pd.concat(data_list, ignore_index=True)
            # print(train_data)

            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            for sensor_id in train_data['sensor'].unique():
                model = loaded_model_dict.get(sensor_id)

                if model is None:
                    logger.warning(f"Model for sensor {sensor_id} not found.")
                    continue

                # Filter data for the current sensor
                df = train_data[train_data['sensor'] == sensor_id]
                df.set_index(['Clock'], inplace=True, drop=True)

                columns_to_drop = ['day', 'weekofyear', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                                   'lag1', 'lag2', 'lag3']
                columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
                df1 = df.drop(columns_to_drop_existing, axis=1, errors='ignore')
                # print(df1)

                # index = df1.index.max()
                # endDate = date.today() + timedelta(days=7)
                # startDate = datetime.today().strftime('%Y-%m-%d')

                future = pd.date_range('2023-11-01', '2023-12-01', freq='1H')
                future_df = pd.DataFrame(index=future)
                future_df['isFuture'] = True
                df1['isFuture'] = False
                df_and_future = pd.concat([df1, future_df])
                df_and_future = create_features(df_and_future)
                df_and_future = add_lags(df_and_future)
                df_and_future['sensor'] = sensor_id

                future_w_features = df_and_future.query('isFuture').copy()

                FEATURES = ['sensor', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2',
                            'lag3']

                with mlflow.start_run():
                    future_w_features['pred'] = model.predict(future_w_features[FEATURES])
                    # print(sensor_id, future_w_features.index, future_w_features['pred'])
                    store_predictions_in_mongodb(sensor_id, future_w_features.index, future_w_features['pred'])
                    # store_actual_val(sensor_id, future_w_features.index, future_w_features['pred'])

                    # Model registry does not work with file store
                    if tracking_url_type_store != "file":
                        # Register the model
                        mlflow.sklearn.log_model(model, "model", registered_model_name=f"{sensor_id}_Model")
                    else:
                        mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())
