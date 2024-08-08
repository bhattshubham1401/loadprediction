import os
import traceback
import os
import traceback
from datetime import datetime, timedelta

import joblib
import pandas as pd

# from src.JdVVNL_Load_Forcastion import logger
# from src.JdVVNL_Load_Forcastion.utils.common import create_features, add_lagsV1, store_predictions_in_mongodb, \
#     holidays_list, data_from_weather_api, sensor_data

import joblib
import pandas as pd
from src.mlProject import logger
from src.mlProject.utils.common import create_features, add_lags, store_predictions_in_mongodb,initialize_mongodb, data_fetchingV1, data_from_weather_apiV1, holidays_list


class PredictionPipeline:
    def __init__(self):
        self.path = 'artifacts/data_ingestion/'
        self.model = 'artifacts/model_trainer/model.joblib'

    def load_model_as_dict(self):
        # Load the model as a dictionary
        return joblib.load(self.model)

    def predict(self):
        try:
            # Load the model as a dictionary
            client, collection = initialize_mongodb("transformed_data")
            count = collection.count_documents({})
            print("no. of id's", count)
            loaded_model_dict = self.load_model_as_dict()

            # Check if the loaded model is a dictionary
            if not isinstance(loaded_model_dict, dict):
                logger.warning("Loaded model is not a dictionary.")
                return

            for i in range(count):
                data = data_fetchingV1(collection, i)
                if not data.empty:
                    model = loaded_model_dict.get(i)

                    if model is None:
                        logger.warning(f"Model for sensor {i} not found.")
                        continue

                    startDate, future_date, end_date = self.get_date(data['Clock'])
                    data.set_index('Clock', inplace=True)

                    holiday_lst = holidays_list(startDate, future_date)
                    future = pd.date_range(end_date, future_date, freq='30min')
                    future_df = pd.DataFrame(index=future)
                    future_df['isFuture'] = True
                    data['isFuture'] = False
                    df_and_future = pd.concat([data, future_df])
                    pd.set_option('display.max_rows', 5000)
                    pd.set_option('display.max_columns', 5000)
                    print("start date", startDate)
                    print("end date", end_date)
                    print("future date", future_date)
                    df_and_future.index.name = 'creation_time'
                    df_and_future.reset_index(['creation_time'], inplace=True)
                    df_and_future['creation_time'] = pd.to_datetime(df_and_future['creation_time'])

                    weather_data = data_from_weather_apiV1(data['site_id'], startDate, future_date)
                    if not weather_data.empty:
                        weather_data['time'] = pd.to_datetime(weather_data['time'])
                        weather_data.set_index('time', inplace=True)
                        weather_data = weather_data[~weather_data.index.duplicated(keep='first')]
                        weather_data = weather_data.resample(rule='30min').ffill()
                        weather_data.reset_index(inplace=True)
                        weather_data['creation_time'] = pd.to_datetime(weather_data['time'])
                        df_and_future['creation_time'] = pd.to_datetime(df_and_future['creation_time'])
                        merged_df = weather_data.merge(df_and_future, on='creation_time', how="inner")

                    # Adding holidays
                    merged_df['holiday'] = merged_df['creation_time'].dt.date.isin(holiday_lst).astype(int)
                    merged_df.set_index(['creation_time'], inplace=True, drop=True)
                    merged_df = add_lags(merged_df)
                    merged_df = create_features(merged_df)
                    future_w_features = merged_df.query('isFuture').copy()
                    # print(future_w_features)
                    # return

                    FEATURES = ['relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m',
                                'holiday', 'lag1', 'lag2', 'lag3', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month',
                                'year']


                    # print(i)

                    future_w_features['pred'] = model.predict(future_w_features[FEATURES])
                    # print(future_w_features['pred'])
                    # future_w_features['pred'].to_csv('merged_df.csv', index=False)
                    store_predictions_in_mongodb(i, future_w_features.index, future_w_features['pred'])

        except Exception as e:
            logger.error(f"Error in Model Prediction: {e}")
            print(traceback.format_exc())

    def get_date(self, startDate_str):
        start_date = startDate_str.min()

        end_date = startDate_str.max()

        # start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        startDate = start_date.replace(hour=0, minute=0, second=0)

        # end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        future_date = end_date + timedelta(days=15)
        future_date = future_date.replace(hour=23, minute=59, second=59)
        return startDate, future_date, end_date