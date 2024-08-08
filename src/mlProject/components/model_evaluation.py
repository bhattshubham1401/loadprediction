import datetime
import traceback
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import create_features, add_lags
from src.mlProject import logger

# class ModelEvaluation:
#     def __init__(self, config: ModelEvaluationConfig):
#         self.config = config

#     def eval_metrics(self, actual, pred):
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         mae = mean_absolute_error(actual, pred)
#         r2 = r2_score(actual, pred)
#         return rmse, mae, r2

#     def load_model_as_dict(self):
#         # Load the model as a dictionary
#         return joblib.load(self.config.model_path)

#     # def actualData(self, data_sensor):
#     #     ''' Dumping Previous month Transformed data into mongo db for Actual vs Predited graph'''
#     #     end_date = datetime.datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#     #     start_date = (end_date - datetime.timedelta(days=end_date.day)).replace(day=1, hour=0, minute=0,
#     #                                                                             second=0, microsecond=0)
#     #
#     #     data_sensor['Clock'] = pd.to_datetime(data_sensor['Clock'])
#     #     last_month_data = data_sensor[(data_sensor['Clock'] >= start_date) & (data_sensor['Clock'] < end_date)]
#     #     store_actual_data(last_month_data)
#     #     return

#     def log_into_mlflow(self):
#         try:
#             data_files = [file for file in os.listdir(self.config.test_data_path) if file.startswith('sensor')]
#             data_list = []
#             for data_file in data_files:
#                 data_sensor = pd.read_csv(os.path.join(self.config.test_data_path, data_file))
#                 # self.actualData(data_sensor)
#                 data_list.append(data_sensor)

#             # Concatenate data for all sensors
#             train_data = pd.concat(data_list, ignore_index=True)
#             # print(train_data)

#             # Load the model as a dictionary
#             loaded_model_dict = self.load_model_as_dict()

#             # Check if the loaded model is a dictionary
#             if not isinstance(loaded_model_dict, dict):
#                 logger.warning("Loaded model is not a dictionary.")
#                 return

#             mlflow.set_registry_uri(self.config.mlflow_uri)
#             tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#             for sensor_id in train_data['sensor'].unique():
#                 model = loaded_model_dict.get(sensor_id)

#                 if model is None:
#                     logger.warning(f"Model for sensor {sensor_id} not found.")
#                     continue

#                 # Filter data for the current sensor
#                 df = train_data[train_data['sensor'] == sensor_id]
#                 df.set_index(['Clock'], inplace=True, drop=True)

#                 columns_to_drop = ['day', 'weekofyear', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
#                                    'lag1', 'lag2', 'lag3']
#                 columns_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
#                 df1 = df.drop(columns_to_drop_existing, axis=1, errors='ignore')
#                 # print(df1)

#                 # index = df1.index.max()
#                 # endDate = date.today() + timedelta(days=7)
#                 # startDate = datetime.today().strftime('%Y-%m-%d')

#                 future = pd.date_range('2023-11-01', '2023-12-01', freq='1H')
#                 future_df = pd.DataFrame(index=future)
#                 future_df['isFuture'] = True
#                 df1['isFuture'] = False
#                 df_and_future = pd.concat([df1, future_df])
#                 df_and_future = create_features(df_and_future)
#                 df_and_future = add_lags(df_and_future)
#                 df_and_future['sensor'] = sensor_id

#                 future_w_features = df_and_future.query('isFuture').copy()

#                 FEATURES = ['sensor', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2',
#                             'lag3']

#                 with mlflow.start_run():
#                     future_w_features['pred'] = model.predict(future_w_features[FEATURES])
#                     # print(sensor_id, future_w_features.index, future_w_features['pred'])
# store_predictions_in_mongodb(sensor_id, future_w_features.index, future_w_features['pred'])
#                     # store_actual_val(sensor_id, future_w_features.index, future_w_features['pred'])

#                     # Model registry does not work with file store
#                     if tracking_url_type_store != "file":
#                         # Register the model
#                         mlflow.sklearn.log_model(model, "model", registered_model_name=f"{sensor_id}_Model")
#                     else:
#                         mlflow.sklearn.log_model(model, "model")

#         except Exception as e:
#             logger.error(f"Error in Model Evaluation: {e}")
#             print(traceback.format_exc())


# import os
from pathlib import Path
# from urllib.parse import urlparse
# from src.mlProject import logger
# import joblib
# import mlflow
# import mlflow.sklearn
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import traceback
from datetime import datetime, timedelta
# from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject.utils.common import save_json, initialize_mongodb, data_fetching


# from src.mlProject.components.data_transformation import create_features, add_lags
# from src.mlProject.utils.common import store_predictions_in_mongodb


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def save_model_as_dict(self, models_dict):
        # Save the model as a dictionary
        joblib.dump(models_dict, self.config.model_path)

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.config.model_path)

    def predict_future_values(self, model, sensor_id, num_periods=24):
        Current_Date = datetime.today()
        NextDay_Date = datetime.today() + timedelta(days=1)

        # Predict for future dates
        future_dates = pd.date_range(start=Current_Date, end=NextDay_Date, freq='H')
        # future_x[0]= 'Clock'

        future_x = create_features(pd.DataFrame({'sensor': [sensor_id] * len(future_dates)}, index=future_dates))
        future_x['Kwh'] = np.nan
        # Include lag features in future_x
        # print(future_x)
        future_x = add_lags(future_x)
        # print(future_x)'is_holiday',
        FEATURES = ['relative_humidity_2m', 'apparent_temperature',
                    'rain',
                    'lag1', 'lag2', 'lag3', 'day', 'hour', 'month', 'year', 'is_holiday']

        X_all = future_x[FEATURES]

        # Predict future values
        future_predictions = model.predict(X_all)

        # Log future predictions to a CSV file
        future_predictions_df = pd.DataFrame({"predicted_kwh": future_predictions}, index=future_dates)
        future_predictions_file_path = f"future_predictions_sensor_{sensor_id}.csv"
        future_predictions_df.to_csv(future_predictions_file_path)

        mlflow.log_artifact(future_predictions_file_path)

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
            # tracking_url_type_store = "http://127.0.0.1:5000"
            print(tracking_url_type_store)

            client, collection = initialize_mongodb("transformed_data")
            count = collection.count_documents({})
            metrix_dict = {}
            for i in range(count):
                data = data_fetching(collection, i)
                if not data.empty:
                    # print(name) 'is_holiday',
                    data.set_index(['Clock'], inplace=True, drop=True)
                    FEATURES = ['relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m',
                                'holiday', 'lag1', 'lag2', 'lag3', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month',
                                'year']

                    TARGET = 'Kwh'

                    X = data[FEATURES]
                    y = data[TARGET]
                    total_length = len(X)
                    # print(X.head())

                    if total_length < 50:
                        print(f"Not enough data for sensor {i}. Skipping...")
                        continue

                    # Calculate dynamic split sizes
                    # train_size = int(0.7 * total_length)
                    # test_size = len(X) - train_size
                    # gap = 24

                    # test_size = 24 * 2  # 7 days * 24 hours/day
                    # gap = 24  # 1 day * 24 hours/day
                    #
                    # total_size = test_size + gap

                    tss = TimeSeriesSplit(n_splits=3)
                    df = data.sort_index()

                    for train_idx, val_idx in tss.split(df):
                        # train_data = data.iloc[train_idx]
                        test_data = df.iloc[val_idx]

                    test_x = test_data[FEATURES]
                    test_y = test_data[TARGET]

                    model = loaded_model_dict.get(i)
                    # print(model)

                    if model is None:
                        logger.warning(f"Model for sensor {i} not found.")
                        continue

                    with mlflow.start_run():
                        predicted_kwh = model.predict(test_x)
                        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_kwh)
                        predicted_df = pd.DataFrame(predicted_kwh, index=test_y.index, columns=["Predicted_Kwh"])

                        # Concatenate the predicted values DataFrame with the actual 'Kwh' values DataFrame
                        result_df = pd.concat([predicted_df, test_y], axis=1)

                        # Print the result

                        # Saving metrics as local
                        scores = {"rmse": rmse, "mae": mae, "r2": r2}
                        metrix_dict[f'{i}'] = scores

                        plt.figure(figsize=(10, 6))
                        plt.plot(result_df.index, result_df['Predicted_Kwh'], label='Predicted Kwh', color='blue')
                        plt.plot(result_df.index, result_df['Kwh'], label='Actual Kwh', color='red')
                        plt.xlabel('Time')
                        plt.ylabel('Kwh')
                        plt.title('Predicted vs Actual Kwh')
                        plt.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        # plt.show()

            save_json(path=Path(self.config.metric_file_name), data=metrix_dict)

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())
        # finally:
        #     client.close()
        #     print("db connection closed")
