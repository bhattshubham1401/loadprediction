import datetime
import json
import os
import traceback
import warnings
from builtins import print

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig

warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from src.mlProject.utils.common import add_lags, create_features, data_from_weather_api, \
    holidays_list, initialize_mongodb, detect_outliers_zscore, sensor_data, drop_leading_zeros, \
    fill_na_with_mean_of_neighbors, store_sensor_data_in_db, plotData

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.le = LabelEncoder()

    def initiate_data_transformation(self):
        try:
            db, client, collection1, collection2 = initialize_mongodb(["sensor", 'transformed_data'])
            df1 = pd.read_parquet(self.config.data_dir)

            startDate = '2024-06-21 00:00:00'
            endDate = '2024-07-31 23:59:59'

            startDate = datetime.datetime.strptime(startDate, '%Y-%m-%d %H:%M:%S')
            endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d %H:%M:%S')
            holiday_lst = holidays_list(startDate, endDate)

            # Label Encoding for 'sensor'
            df1['label_sensor'] = self.le.fit_transform(df1['sensor'])
            sensor_ids = df1['label_sensor'].unique()

            # Create a dictionary to map encoded values to original sensor IDs
            self.sensorDecode(sensor_ids)

            # using groupby function to collect data of each_id
            dframe = df1.groupby('label_sensor')

            for labeled_id, data in dframe:
                # print(labeled_id)
                sensor_df = data
                # data_filepath = os.path.join(self.config.data_file_path, f"raw{labeled_id}.csv")
                # sensor_df.to_csv(data_filepath, mode='w', header=True, index=False)

                outliers = detect_outliers_zscore(sensor_df['Kwh'])
                mean_value = sensor_df['Kwh'].mean()
                sensor_df.loc[outliers, 'Kwh'] = mean_value

                """ Data Validation regarding Voltage and Current """
                filtered_df = sensor_df[
                    ((sensor_df['R_Voltage'] == 0) | (sensor_df['Y_Voltage'] == 0) | (sensor_df['B_Voltage'] == 0)) & (
                            (sensor_df['R_Current'] == 0) | (
                            sensor_df['Y_Current'] == 0) | (sensor_df['B_Current'] == 0))]
                filtered_df['Kwh'] = 0
                sensor_df.loc[sensor_df.index.isin(filtered_df.index), :] = filtered_df

                '''Data Conversion'''
                sensor_df['Clock'] = pd.to_datetime(sensor_df['Clock'])
                sensor_df.set_index(['Clock'], inplace=True, drop=True)
                sensor_id = sensor_df['sensor'].unique()

                '''Resampling dataframe into one-hour interval '''
                dfresample = sensor_df[['Kwh']].resample(rule='30min').sum()

                # checking Lags features
                # fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                # plot_acf(dfresample['Kwh'], lags=48, ax=ax[0])  # Check up to 2 days worth of lags
                # plot_pacf(dfresample['Kwh'], lags=48, ax=ax[1])
                # plt.show()

                dfresample.dropna(subset=['Kwh'], inplace=True)
                dfresample.fillna(value=0, inplace=True)
                dfresample['labeled_id'] = labeled_id
                dfresample['sensor_id'] = sensor_id[0]
                dfresample['holiday'] = 0
                sensor_df.reset_index(['Clock'], inplace=True, drop=True)
                # data_filepath = os.path.join(self.config.data_file_path, f"initial{labeled_id}.csv")
                # dfresample.to_csv(data_filepath, mode='w', header=True, index=False)

                # adding holidays
                for date in holiday_lst:
                    dfresample.loc[f"{date}", 'holiday'] = 1

                dfresample.dropna(inplace=True)
                dfresample = add_lags(dfresample)
                dfresample = create_features(dfresample)
                dfresample.reset_index(inplace=True)
                # plotData(dfresample)
                sen_list = list(dfresample['sensor_id'].unique())
                site_id = sensor_data(sen_list)
                weather_data = data_from_weather_api(site_id['site_id'], startDate, endDate)

                if not weather_data.empty:
                    weather_data['time'] = pd.to_datetime(weather_data['time'])
                    weather_data.set_index('time', inplace=True)
                    weather_data = weather_data[~weather_data.index.duplicated(keep='first')]
                    weather_data = weather_data.resample(rule='30min').ffill()
                    weather_data.reset_index(inplace=True)
                    weather_data['creation_time'] = pd.to_datetime(weather_data['time'])
                    dfresample['creation_time'] = pd.to_datetime(dfresample['Clock'])
                    merged_df = weather_data.merge(dfresample, on='creation_time', how="inner")

                    # Apply the function
                    pd.set_option('display.max_rows', 5000)
                    pd.set_option('display.max_columns', 5000)
                    df_cleaned = drop_leading_zeros(merged_df, 'Kwh')

                    columns_to_fill = ['temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
                                       'wind_speed_10m', 'wind_speed_100m']

                    # Apply the function
                    df_filled = fill_na_with_mean_of_neighbors(df_cleaned, columns_to_fill)
                    columns_to_drop = ['_id','creation_time_iso', 'creation_time', 'time']
                    df_filled.drop(columns=columns_to_drop, inplace=True)

                    store_sensor_data_in_db(db, collection2, labeled_id,  df_filled)

                    # data_filepath = os.path.join(self.config.data_file_path, f"completedata{labeled_id}.csv")
                    # df_filled.to_csv(data_filepath, mode='w', header=True, index=False)
                    # plotData(df_filled)

        except Exception as e:
            print(traceback.format_exc())
            logger.info(f"Error occur in Data Transformation Layer {e}")

        finally:
            client.close()
            print("db connection closed")

    def sensorDecode(self, sensor_ids):
        try:
            decoded_values = self.le.inverse_transform(sensor_ids)
            encoded_to_sensor_mapping = {str(encoded_value): str(original_sensor_id) for
                                         encoded_value, original_sensor_id in
                                         zip(sensor_ids, decoded_values)}

            # Print the mapping
            print("Encoded to Sensor ID Mapping:")
            for encoded_value, original_sensor_id in encoded_to_sensor_mapping.items():
                print(f"Encoded Value {encoded_value} corresponds to Sensor ID {original_sensor_id}")
                # pass
            # Write the mapping to a JSON file
            output_file_path = 'encoded_to_sensor_mapping.json'
            with open(output_file_path, 'w') as file:
                json.dump(encoded_to_sensor_mapping, file)

        except ValueError as e:
            # Handle the case where unknown values are encountered
            print(f"Error decoding sensor IDs: {e}")
