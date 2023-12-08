import os
import traceback
import warnings
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from src.mlProject.utils.common import adfuller_test
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig
import datetime
warnings.filterwarnings("ignore")
import pandas as pd
import seaborn as sns

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from src.mlProject.utils.common import add_lags, create_features, store_actual_data


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self):
        try:
            df1 = pd.read_parquet(self.config.data_dir)
            ''' Storing data im mongo for actual vs predicted '''
            df1['Kwh'] = df1['Kwh'] / 1000

            # Label Encoding for 'sensor'
            le = LabelEncoder()
            df1['sensor'] = le.fit_transform(df1['sensor'])

            sensor_ids = df1['sensor'].unique()

            for sensor_id in sensor_ids:
                sensor_df = df1[df1['sensor'] == sensor_id]

                filtered_df = sensor_df[
                    ((sensor_df['R_Voltage'] == 0) | (sensor_df['Y_Voltage'] == 0) | (sensor_df['B_Voltage'] == 0)) & (
                            (sensor_df['R_Current'] == 0) | (
                            sensor_df['Y_Current'] == 0) | (sensor_df['B_Current'] == 0))]
                filtered_df['Kwh'] = 0

                sensor_df.loc[sensor_df.index.isin(filtered_df.index), :] = filtered_df

                '''Data Conversion'''
                sensor_df['Clock'] = pd.to_datetime(sensor_df['Clock'])
                sensor_df.set_index(['Clock'], inplace=True, drop=True)
                sensor_df = sensor_df[sensor_df.index >= '2022-11-18 00:00:00']

                # sensor_df['Kwh'].plot(figsize=(10, 5), color=color_pal[4], ms=1, lw=1,
                #                                title='Future Predictions')
                # plt.show()

                '''Resampling dataframe into one-hour interval '''

                dfresample = sensor_df[['Kwh']].resample(rule='1H').sum()
                seasonal = seasonal_decompose(dfresample['Kwh'], model='additive', period=1)
                dfresample = pd.concat([seasonal.observed, seasonal.seasonal, seasonal.trend, seasonal.resid], axis=1)
                dfresample.columns = ["Kwh", "seasonal", 'trend', 'resid']
                dfresample['Kwh'] = dfresample['Kwh']
                dfresample['Kwh'] = dfresample['Kwh'] - dfresample['seasonal']
                dfresample.dropna(subset=['Kwh'], inplace=True)
                dfresample['sensor'] = sensor_id
                sensor_df.reset_index(['Clock'], inplace=True, drop=True)
                # self.adfuller_test(dfresample['Kwh'], sensor_id)

                dfresample = add_lags(dfresample)
                dfresample = create_features(dfresample)

                FEATURES = ['sensor', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2',
                            'lag3']
                TARGET = ['Kwh']

                X_all = dfresample[FEATURES]

                y_all = dfresample[TARGET]
                X_all.reset_index(inplace=True)
                y_all.reset_index(inplace=True)
                dfresample.reset_index(inplace=True)

                train_data_filepath = os.path.join(self.config.root_dir, f"train_data_sensor_{sensor_id}.csv")
                test_data_filepath = os.path.join(self.config.root_dir, f"test_data_sensor_{sensor_id}.csv")
                data_filepath = os.path.join(self.config.data_file_path, f"sensor_{sensor_id}.csv")
                # Save data to separate train and test files for each sensor
                X_all.to_csv(train_data_filepath, mode='w', header=True, index=False)
                y_all.to_csv(test_data_filepath, mode='w', header=True, index=False)
                dfresample.to_csv(data_filepath, mode='w', header=True, index=False)

                # ''' Dumping Previous month Transformed data into mongo db for Actual vs Predited graph'''
                end_date = datetime.datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                start_date = (end_date - datetime.timedelta(days=end_date.day)).replace(day=1, hour=0, minute=0,
                                                                                         second=0, microsecond=0)

                dfresample['Clock'] = pd.to_datetime(dfresample['Clock'])
                last_month_data = dfresample[(dfresample['Clock'] >= start_date) & (dfresample['Clock'] < end_date)]
                store_actual_data(last_month_data)

        except Exception as e:
            print(traceback.format_exc())
            logger.info(f"Error occur in Data Transformation Layer {e}")



