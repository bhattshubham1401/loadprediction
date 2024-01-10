import json
import os
import traceback
import warnings
from datetime import date as datetime_date
from sklearn.preprocessing import LabelEncoder
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import  TimeSeriesSplit
from src.mlProject.utils.common import add_lags, create_features ,\
    holidays_list, store_sensor_data_in_db, initialize_mongodb, uom #, data_from_weather_api 

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.le = LabelEncoder()

    def initiate_data_transformation(self):
        try:
            client, collection5, collection6, collection7 = initialize_mongodb(["sensor", 'train', 'test'])
            df1 = pd.read_parquet(self.config.data_dir)

            # Label Encoding for 'sensor'
            df1['label_sensor'] = self.le.fit_transform(df1['sensor'])
            sensor_ids = df1['label_sensor'].unique()

            # Create a dictionary to map encoded values to original sensor IDs
            self.sensorDecode(sensor_ids)

            # using groupby function to collect data of each_id
            dframe = df1.groupby('label_sensor')

            # UOM and sensor_name
            df_uom = uom()
            
            # holidays data
            start_date = datetime_date(2022, 11, 30)
            end_date = datetime_date(2023, 11, 30)
            holiday_lst = holidays_list(start_date, end_date)
            
            # weather data
            weather_data = pd.read_csv(self.config.weather_data_file_path)
            weather_data['Clock'] = pd.to_datetime(weather_data['Clock'])
            weather_data.set_index(['Clock'],inplace=True, drop=True)
            weather_data.drop(['temp'],axis=1,inplace=True)
            weather_data.drop(['precipitation','apparent_temp'],axis=1,inplace=True)
            # weather data from API
            # weather_data = data_from_weather_api()
            
            for labeled_id, data in dframe:
                sensor_df = data

                """ Data Validation regarding Voltage and Current """
                filtered_df = sensor_df[((sensor_df['R_Voltage'] == 0) | (sensor_df['Y_Voltage'] == 0) | (sensor_df['B_Voltage'] == 0)) & (
                            (sensor_df['R_Current'] == 0) | (
                            sensor_df['Y_Current'] == 0) | (sensor_df['B_Current'] == 0))]
                filtered_df['Kwh'] = 0
                sensor_df.loc[sensor_df.index.isin(filtered_df.index), :] = filtered_df

                '''Data Conversion'''
                sensor_df['Clock'] = pd.to_datetime(sensor_df['Clock'])
                sensor_df.set_index(['Clock'], inplace=True, drop=True)

                sensor_df = sensor_df[sensor_df.index >= '2022-11-18 00:00:00']
                sensor_id = sensor_df['sensor'].unique()
                uom_value = df_uom[df_uom['uuid'] == sensor_id[0]]['UOM'].values[0]
                sensor_name = df_uom[df_uom['uuid'] == sensor_id[0]]['sensorName'].values[0]
                if uom_value == 'MWH':
                    sensor_df['Kwh'] = sensor_df['Kwh'] / 1000

                '''Resampling dataframe into one-hour interval '''

                dfresample = sensor_df[['Kwh']].resample(rule='1H').sum()

                dfresample['labeled_id'] = labeled_id
                dfresample['sensor_id'] = sensor_id[0]
                
                # adding weather data
                dfresample=pd.merge(dfresample,weather_data, on="Clock")

                # adding holidays
                # dfresample['holiday'] = dfresample['Clock'].dt.date.isin(holiday_lst).astype(int)   
                dfresample['holiday'] = np.isin(dfresample.index.date, holiday_lst).astype(int)
                
                # adding lags and features
                dfresample = add_lags(dfresample)
                dfresample = create_features(dfresample)
                
                dfresample.reset_index(inplace=True)

                tss = TimeSeriesSplit(n_splits=5, test_size=24 * 30 * 1, gap=24)
                df = dfresample.sort_index()
                # df.dropna(subset=['Kwh'], inplace=True)
                for train_idx, val_idx in tss.split(df):
                    train_data = df.iloc[train_idx]
                    test_data = df.iloc[val_idx]

                # storing each sensor_id data in db for testing and training
                store_sensor_data_in_db(collection5, collection6, collection7, labeled_id, sensor_name,
                                         dfresample, train_data, test_data)
            logger.info(f"count of id's in db {collection5.count_documents({})}")

        except Exception as e:
            # print(traceback.format_exc())
            logger.info(f"Error occur in Data Transformation Layer {e}")

        finally:
            client.close()
            logger.info("db connection closed")

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