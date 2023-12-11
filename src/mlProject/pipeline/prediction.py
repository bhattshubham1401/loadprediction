import os
import traceback
import datetime
import joblib
import pandas as pd

from src.mlProject import logger
from src.mlProject.utils.common import create_features, add_lags, store_predictions_in_mongodb, store_actual_data


class PredictionPipeline:
    def __init__(self):
        self.path = 'artifacts/data_ingestion/'
        self.model = 'artifacts/model_trainer/model.joblib'

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.model)

    def predict(self):
        try:
            data_files = [file for file in os.listdir(self.path) if file.startswith('sensor')]
            data_list = []

            for data_file in data_files:
                data_sensor = pd.read_csv(os.path.join(self.path, data_file))
                data_list.append(data_sensor)
                # self.actualData(data_sensor)

            # Concatenate data for all sensors
            train_data = pd.concat(data_list, ignore_index=True)

            # Load the model as a dictionary
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

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
                # endDate = datetime.date.today() + datetime.timedelta(days=7)
                # startDate = datetime.today().strftime('%Y-%m-%d')

                future = pd.date_range(start='2023-12-01',end='2023-12-31', freq='1H')
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

                future_w_features['pred'] = model.predict(future_w_features[FEATURES])
                store_predictions_in_mongodb(sensor_id, future_w_features.index, future_w_features['pred'])

            return

        except Exception as e:
            logger.error(f"Error in Model Evaluation: {e}")
            print(traceback.format_exc())
