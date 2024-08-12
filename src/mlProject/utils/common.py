'''functionality that we are use in our code'''

import json
import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Any

import holidays
import joblib
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from dotenv import load_dotenv
from ensure import ensure_annotations
from matplotlib import pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import TimeSeriesSplit

from src.mlProject import logger

# from src.mlProject.components.data_transformation import sensorDecode

load_dotenv()


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args\\\\\\\\\\\\\\
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB
       memory_usage = hourly_data.memory_usage(deep=True).sum() / (1024 ** 2)
    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def convert_datetime(input_datetime):
    parsed_datetime = datetime.strptime(input_datetime, '%Y-%m-%dT%H:%M')
    formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_datetime


# @ensure_annotations
# def get_mongoData():
#     ''' calling DB configuration '''
#
#     logger.info("calling DB configuration")
#     db = os.getenv("db")
#     host = os.getenv("host")
#     port = os.getenv("port")
#     collection = os.getenv("collection")
#
#     MONGO_URL = f"mongodb://{host}:{port}"
#
#     ''' Read data from DB'''
#
#     '''Writing logs'''
#     logger.info("Reading data from Mongo DB")
#
#     '''Exception Handling'''
#
#     try:
#         client = MongoClient(MONGO_URL)
#         db1 = client[db]
#         collection = db1[collection]
#
#         data = collection.find({})
#
#         columns = ['sensor', 'Clock', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
#                    'B_Current', 'A', 'BlockEnergy-WhExp', 'B', 'C', 'D', 'BlockEnergy-VAhExp',
#                    'Kwh', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp']
#
#         datalist = [(entry['sensor_id'], entry['raw_data']) for entry in data]
#         df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)
#
#         '''Dropping Columns'''
#         df = df.drop(
#             ['BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
#              'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
#         pd.set_option('display.max_columns', None)
#
#         # print("===============DataType Conversion==================")
#         df['Clock'] = pd.to_datetime(df['Clock'])
#         df['Kwh'] = df['Kwh'].astype(float)
#         df['R_Voltage'] = df['R_Voltage'].astype(float)
#         df['Y_Voltage'] = df['Y_Voltage'].astype(float)
#         df['B_Voltage'] = df['B_Voltage'].astype(float)
#         df['R_Current'] = df['R_Current'].astype(float)
#         df['Y_Current'] = df['Y_Current'].astype(float)
#         df['B_Current'] = df['B_Current'].astype(float)
#         return df
#
#     except Exception as e:
#         logger.info(f"Error occurs =========== {e}")

''' Fetching Data from mongo DB through API'''


@ensure_annotations
def get_data_from_api_query():
    ''' '''
    logger.info("fetching data")
    try:
        lst = ['66755ea75d9656.09451425',
               '66792887ab2bf0.01972524',
               '66798b07a176f7.28421426',
               '66796c7d38ef95.94782159',
               '667975ae47a9d3.45611637',
               '66798c998af0a4.39772704',
               '66798d6f3b99a8.11022930',
               '66798dd85a2067.34235472',
               '66798e4245c0d7.47939284',
               '667bd42c040dd4.71905290',
               '667be335c9c907.26863078',
               '66798f9a010539.13786166',
               '66798d04b74f65.00513254',
               '667be7240c74a4.64013414',
               '66798f359b8a34.89774241',
               '66799027e46756.51131984',
               '667be8df45fab9.64137172',
               '667be9cb3031e1.24562866',
               '667989e30478b9.04540881',
               '667beac5d62785.79876238',
               '667c0ece2d5d57.23218009',
               '667be57d43ddc1.43514258',
               '667be670c7f740.52758876',
               '667bfdfeab6d34.51534194',
               '667bff2df379b5.53081304',
               '667bffb3b3ae66.16138506',
               '667c05ad3595d9.00092026',
               '667e677c217958.45562798',
               '667e677c2549e8.85289343',
               '667c07a988c680.36467497',
               '667e677c2bddd7.76320522',
               '667c0867052b10.12209224',
               '667e58d91b6379.53203432',
               '667c09221b1994.79645670',
               '667e677c1e25e7.27858012',
               '667c0caff0c527.66621614',
               '667e58d9166f67.75643219',
               '667e58d91402f9.55869379',
               '667c0d7c1a2c25.42171062',
               '667e58d918dfe6.23747237',
               '667c09c03bb026.40883695',
               '667c0f783366a2.15185331',
               '667e677c2f4e53.85361321',
               '667e58d8e67dc9.69173999',
               '667e58d8e8a9f3.26329150',
               '667e58d8eaefe0.18391362',
               '667e58d8efebc2.38937885',
               '667e58d9004242.97860565',
               '667e58d904aea6.48830596',
               '667e58d8f20555.60582824',
               '667d31828576d8.46037940',
               '667d320e166875.30973434',
               '667d3293d6fcc2.53026792',
               '667c065f45a327.55067013',
               '667d2fe49edd63.94185560',
               '667d2d726d34d9.80543124',
               '667d2ed3431ed1.79929882',
               '667d2b22923911.57953310',
               '667e677c722cd4.54466988',
               '667c1208be16e3.98383881',
               '667c485f88cb41.11683168',
               '667d1cd9150f44.31238978',
               '667d1f47aca158.41077537',
               '667c12bb905a58.52710727',
               '667d1921657499.98433906',
               '667e6dd7dd4a69.18470849',
               '667c1332cd8232.01161681',
               '667c14616ac802.18010687',
               '667c03d72b1502.67552912',
               '667c15626a3e40.50715063',
               '667e677c3950d9.96416684',
               '667e677c3cbac5.50389286',
               '667d15b1ac09a7.11635501',
               '667e677c408dd7.97426812',
               '667e677c587561.43422097',
               '667d17335293c2.93969318',
               '667e6dd8602869.69317036',
               '667e6dd8418eb0.29956026',
               '667e6dd848adf1.98125995',
               '667e6dd84c56d1.57393902',
               '667e6dd85d3ca6.45572149',
               '667e6dd8632305.25075935',
               '667cfde9845216.22492904',
               '667c1947cf8119.42008676',
               '667e677c6978f5.52670606',
               '667c1864225670.88486374'
               ]

        l1 = []
        # toDate = datetime(2024, 1, 31, 23, 59, 59)

        url = "http://jpdclmdm.radius-ami.com:8850/gtDta"
        headers = {
            "Content-Type": "application/json",
            "token": "3r+oBfrZkdVDOzFLmIYfl2PZyNJ74am9A4pGupWY+Sf8wIGArIz+awUEdWSLBPidh1B2nKaaAPn7m64WhjKfdhu1XLGo4gCcniOpmaSSDwRPzcXTJVwA1Am6QAIud0Yp13fujXWcariJ5nJeKNGtO/nmRS4dFYu0f4ZHEON/xv5XDBiKWmAeAmb5v7p4jB1+PHxpfSUfOX8rKIfGV/APO0bCc4KVIzgJbgtKw3nNcHh3YIce7oY3e/BjkCUGCdNOUI7kFjcZqT0OT5PEkUTNC5lJrmWB03uNHWrGe/LDq9+5hPUrc9x44jEOoKiwMHbd6yfTW+1qZXmdScXOZIQdcVQK23u2zeRGrGntmrdXXoCzJvbbJnFi0OtAjcfOemDVJL8I3mzZLGTJXL5UbcT8WQ==",
            "api_gateway": "AMI",
            "APIAgent": "shubh"
        }

        for sensor in lst:
            params = {
                "sensor_id": sensor,
                "type": "LP",
                "count": "0",
                "sub_type": "FDR",
                "r_count": "5000"
            }
            response = requests.post(url, json=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            print(data)
            l1.append(data['DATA'])

        columns = ['sensor', 'Clock', 'R_Current', 'Y_Current', 'B_Current', 'R_Voltage', 'Y_Voltage', 'B_Voltage',
                   'Kwh', 'BlockEnergy-WhExp', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ2', 'BlockEnergy-VArhQ3',
                   'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp',
                   'BlockEnergy-VAhExp', 'MeterHealthIndicator']

        datalist = [(entry['sensor_id'], entry['raw_data']) for i in range(len(l1)) for entry in l1[i]]
        # print(datalist)
        df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)
        df = df.drop(['BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ2', 'BlockEnergy-VArhQ3',
                      'BlockEnergy-VArhQ4', 'MeterHealthIndicator', 'BlockEnergy-WhExp', 'BlockEnergy-VAhExp',
                      'BlockEnergy-VAhImp', ], axis=1)
        # df = df.drop([
        #     'BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
        #     'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
        pd.set_option('display.max_columns', None)

        df['Clock'] = pd.to_datetime(df['Clock'])
        df['sensor'] = df['sensor'].astype(str)
        df['R_Voltage'] = df['R_Voltage'].astype(float)
        df['Y_Voltage'] = df['Y_Voltage'].astype(float)
        df['B_Voltage'] = df['B_Voltage'].astype(float)
        df['R_Current'] = df['R_Current'].astype(float)
        df['Y_Current'] = df['Y_Current'].astype(float)
        df['B_Current'] = df['B_Current'].astype(float)
        df['Kwh'] = df['Kwh'].astype(float)
        # print(df.tail())
        return df

    except Exception as e:
        logger.info(f"Error occurs =========== {e}")


@ensure_annotations
def load_file():
    file = os.getenv("filename")
    return file


@ensure_annotations
def plotData(df1):
    plt.figure(figsize=(10, 6))
    plt.scatter(df1['Clock'], df1['Kwh'], label='Actual')
    plt.xlabel('Clock ')
    plt.ylabel('Kwh')
    plt.legend()
    plt.show()
    return

    # Line plot
    # sns.lineplot(x='x_column', y='y_column', data=data)
    # plt.show()
    #
    # # Histogram
    # sns.histplot(data['numeric_column'], bins=10)
    # plt.show()
    #
    # # Box plot
    # sns.boxplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Bar plot
    # sns.barplot(x='category_column', y='numeric_column', data=data)
    # plt.show()
    #
    # # Pair plot (for exploring relationships between multiple variables)
    # sns.pairplot(data)
    # plt.show()


@ensure_annotations
def sliderPlot(df1):
    fig = px.line(df1, x=df1['meter_timestamp'], y=df1['KWh'])
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )

    )
    fig.show()
    return


@ensure_annotations
def store_predictions_in_mongodb(sensor_id, dates, predictions):
    try:
        logger.info("Calling DB configuration")

        # Load the labeled_to_original_mapping from a JSON file
        with open("encoded_to_sensor_mapping.json", "r") as file:
            labeled_to_original_mapping = json.load(file)

        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection6")

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]

        unique_dates = sorted(set(dates.date))[:-1]

        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            original_sensor_id = labeled_to_original_mapping.get(str(sensor_id), str(sensor_id))
            document_id = f"{original_sensor_id}_{date_str}"

            data = {
                "_id": document_id,
                "sensor_id": original_sensor_id,
                "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "millisecond": int(datetime.now().timestamp() * 1000),
                "data": {}
            }

            # Filter predictions for the current date
            date_predictions = predictions[dates.date == date]

            # Populate the 'data' dictionary with hourly predictions
            for i, prediction in enumerate(date_predictions):
                prediction_float = round(float(prediction), 4)
                data["data"][str(i)] = {
                    "pre_kwh": prediction_float,
                    "pre_current": 0.0,
                    "pre_load": 0.0,
                    "act_kwh": 0.0,
                    "act_load": 0.0
                }

            data_dict = {key: float(value) if isinstance(value, (float, np.integer, np.floating)) else value
                         for key, value in data.items()}

            # Insert data into MongoDB
            collection.insert_one(data_dict)

        client.close()
        logger.info("Data stored successfully")

    except Exception as e:
        print(e)


def create_features(hourly_data):
    hourly_data = hourly_data.copy()

    # Check if the index is in datetime format
    if not isinstance(hourly_data.index, pd.DatetimeIndex):
        hourly_data.index = pd.to_datetime(hourly_data.index)

    hourly_data['day'] = hourly_data.index.day
    hourly_data['hour'] = hourly_data.index.hour
    hourly_data['month'] = hourly_data.index.month
    hourly_data['dayofweek'] = hourly_data.index.dayofweek
    hourly_data['quarter'] = hourly_data.index.quarter
    hourly_data['dayofyear'] = hourly_data.index.dayofyear
    hourly_data['weekofyear'] = hourly_data.index.isocalendar().week
    hourly_data['year'] = hourly_data.index.year
    return hourly_data


@ensure_annotations
def add_lags(df):
    target_map = df['Kwh'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('1D')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('7D')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('15D')).map(target_map)

    return df


# @ensure_annotations
# def rolling_statistics(self, data):
#     MA = data.rolling(window=24).mean()
#     MSTD = data.rolling(window=24).std()
#     plt.figure(figsize=(15, 5))
#     orig = plt.plot(data, color='black', label='Original')
#     mean = plt.plot(MA, color='red', label='MA')
#     std = plt.plot(MSTD, color='yellow', label='MSTD')
#     plt.legend(loc='best')
#     plt.title("Rolling Mean and standard Deviation")
#     plt.show()


@ensure_annotations
def adfuller_test(self, data, sensor_id):
    print("Result od adifuller test:")
    # dftest = adfuller(data, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Stat', 'p-value', 'lags used', 'np of observation used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Values(%s)' % key] = value
    # print(f"The sensor id {sensor_id}-{dfoutput}")


# @ensure_annotations
# def store_actual_data(data):
#     # print(data[['Clock', 'Kwh', 'sensor']])
#     # return
#
#     try:
#         logger.info("calling DB configuration")
#         db = os.getenv("db")
#         host = os.getenv("host")
#         port = os.getenv("port")
#         collection_name = os.getenv("collection3")
#
#         MONGO_URL = f"mongodb://{host}:{port}"
#
#         labeled_to_original_mapping = {
#             0: "5f718c439c7a78.65267835",
#             1: "62a9920f75c931.62399458",
#         }
#
#         client = MongoClient(MONGO_URL)
#         db1 = client[db]
#         collection = db1[collection_name]
#
#         # Group data by date
#         grouped_data = data.groupby(data['Clock'].dt.date)
#
#         for date, group in grouped_data:
#             date_str = date.strftime('%Y-%m-%d')
#             document_id = f"{labeled_to_original_mapping.get(group['sensor'].iloc[0], group['sensor'].iloc[0])}_{date_str}"
#
#             document = {
#                 "_id": document_id,
#                 "sensor_id": labeled_to_original_mapping.get(group['sensor'].iloc[0], group['sensor'].iloc[0]),
#                 "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#                 "millisecond": int(datetime.now().timestamp() * 1000),
#                 "data": {}
#             }
#
#             # Populate the 'data' dictionary with hourly predictions
#             for _, row in group.iterrows():
#                 actuals = round(float(row['Kwh']), 4)
#                 hour = row['Clock'].hour
#                 data_key = str(hour)
#                 data_value = {
#                     "act_kwh": actuals
#                 }
#                 document["data"][data_key] = data_value
#
#             # Insert data into MongoDB
#             collection.insert_one(document)
#
#         client.close()
#
#     except Exception as e:
#         print(e)

@ensure_annotations
def data_from_weather_api(site, startDate, endDate):
    logger.info("Weather data fetching")
    try:
        start_date = startDate.strftime('%Y-%m-%d %H:%M:%S')
        end_date = endDate.strftime('%Y-%m-%d %H:%M:%S')
        site_array = np.array(site, dtype=object)
        site = str(site_array[0])
        print("Start Date:", start_date)
        print("End Date:", end_date)
        print("Site ID:", site)

        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection3")
        print("Collection:", collection_name)

        MONGO_URL = f"mongodb://{host}:{port}"

        ''' Read data from DB'''

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]
        documents = []
        query = collection.find({
            "time": {
                "$gte": start_date,
                "$lte": end_date
            },
            "site_id": site
        })
        for doc in query:
            documents.append(doc)
            # print(documents)
        try:

            df = pd.DataFrame(documents)
            return df
        except Exception as e:
            print(e)
    except Exception as e:
        print("Error:", e)


def holidays_list(start_date_str, end_date_str):
    logger.info("Generating holidays list")
    try:
        # start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        # end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        start_date = start_date_str.date()
        end_date = end_date_str.date()

        holiday_list = []

        # Get the holiday dates in India for the specified year
        india_holidays = holidays.CountryHoliday('India', years=start_date.year)

        # Iterate through each date from start_date to end_date
        current_date = start_date
        while current_date <= end_date:
            # Check if the current date is a holiday in India or a Sunday
            if current_date in india_holidays or current_date.weekday() == 6:
                holiday_list.append(current_date)
            current_date += timedelta(days=1)

        return holiday_list

    except Exception as e:
        logger.error(f"Error in holidays_list: {e}")
        return None


# def holidays_list(start_date, end_date):
#     logger.info("holidays list")
#     try:
#         holiday_list = []

#         def is_holiday(single_date):
#             year = single_date.year
#             country_holidays = holidays.CountryHoliday('India', years=year)
#             return single_date in country_holidays or single_date.weekday() == 6  # Sunday is represented by 6

#         date_list = [
#             (single_date, holidays.CountryHoliday('India', years=single_date.year).get(single_date))
#             for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1))
#             if is_holiday(single_date)]
#         for date, name in date_list:
#             # print(f"{date}: {name}")
#             holiday_list.append(date)
#         return holiday_list
#     except Exception as e:
#         print(e)


@ensure_annotations
def dataset_count():
    try:
        logger.info("calling DB configuration for data")
        host = os.getenv("host")
        port = os.getenv("port")
        db = os.getenv("db")
        collection_name = os.getenv("collection2")

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection = db1[collection_name]
        # db1.collection.drop()
        count = collection.count_documents({})
        client.close()

        return count

    except Exception as e:
        print(e)


@ensure_annotations
def initialize_mongodb(collection_name):
    try:
        logger.info("DB connection established")
        host = os.getenv("host")
        port = os.getenv("port")
        db = os.getenv("db")
        collection_name1 = os.getenv("collection1")
        collection_name2 = os.getenv("collection2")
        collection_name4 = os.getenv("collection4")
        # print(db)
        # print(collection_name1)
        # print(collection_name2)
        # print(collection_name3)

        mongo_url = f"mongodb://{host}:{port}"
        client = MongoClient(mongo_url)
        db1 = client[db]
        collection1 = db1[collection_name1]
        collection2 = db1[collection_name2]
        collection4 = db1[collection_name4]

        if type(collection_name) == list:
            return db1, client, collection1, collection2
        elif collection_name == "sensor":
            return client, collection1
        elif collection_name == "transformed_data":
            return client, collection2
        elif collection_name == "transformTest":
            return client, collection4
        # elif collection_name == "test":
        #     return client, collection3
    except Exception as e:
        print(e)


@ensure_annotations
# def store_sensor_data_in_db(sensor_id, df,collection, db, sensorName)
def store_sensor_data_in_db(db, collection, sensor_id, dfresample):
    try:
        if sensor_id == 0:
            logger.info("Data found")
            db.drop_collection(collection.name)

        sensor_data = dfresample.to_dict(orient='records')
        id_data1 = {"sensor_id": sensor_id, "data": sensor_data}
        collection.insert_one(id_data1)
        logger.info("Data stored")

    except Exception as e:
        print(e)


@ensure_annotations
def data_fetching(collection, i):
    try:
        data_list = list(collection.find({"sensor_id": i}, {'_id': 0, 'data': 1}))
        # print(data_list)
        df = pd.DataFrame(data_list[0]['data'])
        # sensor_name = data_list[0]['sensor_name']
        return df
    except Exception as e:
        print(e)


@ensure_annotations
def data_fetchingV1(collection, i):
    try:
        data_list = list(collection.find({"sensor_id": i}, {'_id': 0, 'data.Clock': 1, 'data.labeled_id': 1, 'data.site_id': 1, 'data.sensor_id': 1, 'data.Kwh': 1}))
        # print(data_list)
        df = pd.DataFrame(data_list[0]['data'])
        # sensor_name = data_list[0]['sensor_name']
        return df
    except Exception as e:
        print(e)


@ensure_annotations
def uom():
    try:
        sensor_ids = [
            '5f718b613291c7.03696209', '5f718c439c7a78.65267835',
            '614366bce31a86.78825897', '6148740eea9db0.29702291', '625fb44c5fb514.98107900',
            '625fb9e020ff31.33961816', '6260fd4351f892.69790282', '627cd4815f2381.31981050',
            '629094ee5fdff4.43505210', '62aad7f5c65185.80723547', '62b15dfee341d1.73837476',
            '62b595eabd9df4.71374208', '6349368c306542.16235883', '634e7c43038801.39310596',
            '6399a18b1488b8.07706749', '63a4195534d625.00718490', '63a4272631f153.67811394',
            '63aa9161b9e7e1.16208626', '63ca403ccd66f3.47133508', '62a9920f75c931.62399458'
        ]

        url = "https://multipoint.myxenius.com/Sensor_newHelper/getDataApi"
        params = {

            'sql': "SELECT id AS uuid, name AS sensorName, CASE WHEN grid_billing_type IS NOT NULL THEN grid_billing_type ELSE 'UOM' END AS uom FROM sensor WHERE id IN ({}) ORDER BY name".format(
                ','.join(f"'{sid}'" for sid in sensor_ids)),
            'type': 'query'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        sensor_list = [{'uuid': item['uuid'], 'sensorName': item['sensorName'], "UOM": item['uom']} for item in
                       data['resource']]
        df = pd.DataFrame(sensor_list)
        logger.info("UOM values")
        return df


    except Exception as e:
        print(e)


@ensure_annotations
def test_train_split(df1):
    try:
        tss = TimeSeriesSplit(n_splits=5, test_size=24 * 30 * 1, gap=24)
        df = df1.sort_index()
        # df.dropna(subset=['Kwh'], inplace=True)
        for train_idx, val_idx in tss.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[val_idx]
            return train_data, test_data
    except Exception as e:
        print(e)


@ensure_annotations
def store_test_data(df, id, db, collection):
    try:
        collection_name = os.getenv("collection3")
        # collection = db[collection_name]
        if id == 0:
            db.drop_collection(collection_name)
            logger.info("deleted found test_data")

        data1 = df.to_dict(orient='records')
        id_data = {"sensor_id": id, "data": data1}

        collection.insert_one(id_data)
    except Exception as e:
        print(e)


@ensure_annotations
def test_data_fetching(collection, i):
    try:
        data_list = list(collection.find({"sensor_id": i}, {'_id': 0, "sensor_name": 1, 'data': 1}))
        df = pd.DataFrame(data_list[0]['data'])
        sensor_name = data_list[0]['sensor_name']
        return df, sensor_name
    except Exception as e:
        print(e)


@ensure_annotations
def detect_outliers_zscore(dataframe, threshold=3):
    from scipy.stats import zscore
    z_scores = np.abs(zscore(dataframe))
    return (z_scores > threshold)


@ensure_annotations
def sensor_data(id_lst):
    try:
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection1")
        MONGO_URL = f"mongodb://{host}:{port}"

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]
        data_list = []
        # for id in id_lst:
        value = list(collection.find(
            {"id": {'$in': id_lst}},
            {"meter_ct_mf": 1, "UOM": 1, "meter_MWh_mf": 1, "site_id": 1, "asset_id": 1,
             "sensor_id": "$parent_sensor_id"}
        ))
        # Use extend instead of append to flatten the list
        data_list.extend(value)
        df = pd.DataFrame(data_list)
        return df
    except Exception as e:
        print(e)


@ensure_annotations
def drop_leading_zeros(df, column):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in the DataFrame")

    # Check if the column contains only zeros or NaN values
    if df[column].replace(0, pd.NA).dropna().empty:
        print(f"Column '{column}' contains only zeros or NaN values")
        return df  # or return an empty DataFrame: return pd.DataFrame(columns=df.columns)

    # Find the first non-zero index
    first_non_zero_idx = df[column].ne(0).idxmax()

    # Return the DataFrame from the first non-zero index onward
    return df.loc[first_non_zero_idx:].reset_index(drop=True)


@ensure_annotations
def fill_na_with_mean_of_neighbors(df, columns):
    for column in columns:
        for i in range(1, len(df) - 1):
            if pd.isna(df.at[i, column]):
                prev_val = df.at[i - 1, column]
                next_val = df.at[i + 1, column]
                if not pd.isna(prev_val) and not pd.isna(next_val):
                    df.at[i, column] = (prev_val + next_val) / 2
    return df


@ensure_annotations
def data_from_weather_apiV1(site, startDate, endDate):
    logger.info("Weather data fetching")
    try:
        start_date = startDate.strftime('%Y-%m-%d %H:%M:%S')
        end_date = endDate.strftime('%Y-%m-%d %H:%M:%S')
        site_array = np.array(site, dtype=object)
        site = str(site_array[0])
        print("Start Date:", start_date)
        print("End Date:", end_date)
        print("Site ID:", site)

        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection3")
        print("Collection:", collection_name)

        MONGO_URL = f"mongodb://{host}:{port}"

        ''' Read data from DB'''

        '''Writing logs'''
        logger.info("Reading data from Mongo DB")

        '''Exception Handling'''
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]
        documents = []
        query = collection.find({
            "time": {
                "$gte": start_date,
                "$lte": end_date
            },
            "site_id": site
        })
        for doc in query:
            documents.append(doc)
            # print(documents)
        try:

            df = pd.DataFrame(documents)
            return df
        except Exception as e:
            print(e)
    except Exception as e:
        print("Error:", e)
