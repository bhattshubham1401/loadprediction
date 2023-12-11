'''functionality that we are use in our code'''

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from dotenv import load_dotenv
from ensure import ensure_annotations
from pymongo import MongoClient

# from statsmodels.tsa.stattools import adfuller

from src.mlProject import logger

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


@ensure_annotations
def get_mongoData():
    ''' calling DB configuration '''

    logger.info("calling DB configuration")
    db = os.getenv("db")
    host = os.getenv("host")
    port = os.getenv("port")
    collection = os.getenv("collection")

    MONGO_URL = f"mongodb://{host}:{port}"

    ''' Read data from DB'''

    '''Writing logs'''
    logger.info("Reading data from Mongo DB")

    '''Exception Handling'''

    try:
        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection]

        data = collection.find({})

        columns = ['sensor', 'Clock', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
                   'B_Current', 'A', 'BlockEnergy-WhExp', 'B', 'C', 'D', 'BlockEnergy-VAhExp',
                   'Kwh', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp']

        datalist = [(entry['sensor_id'], entry['raw_data']) for entry in data]
        df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)

        '''Dropping Columns'''
        df = df.drop(
            ['BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
             'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
        pd.set_option('display.max_columns', None)

        # print("===============DataType Conversion==================")
        df['Clock'] = pd.to_datetime(df['Clock'])
        df['Kwh'] = df['Kwh'].astype(float)
        df['R_Voltage'] = df['R_Voltage'].astype(float)
        df['Y_Voltage'] = df['Y_Voltage'].astype(float)
        df['B_Voltage'] = df['B_Voltage'].astype(float)
        df['R_Current'] = df['R_Current'].astype(float)
        df['Y_Current'] = df['Y_Current'].astype(float)
        df['B_Current'] = df['B_Current'].astype(float)
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
    plt.scatter(df1['KWh'], df1['cumm_PF'], label='Actual')
    plt.xlabel('KWh ')
    plt.ylabel('cumm_PF')
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
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection2")

        MONGO_URL = f"mongodb://{host}:{port}"

        labeled_to_original_mapping = {
            0: "5f718c439c7a78.65267835",
            1: "62a9920f75c931.62399458",
        }

        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]

        unique_dates = set(dates.date)
        unique_dates = sorted(unique_dates)[:-1]

        for date in unique_dates:
            date_str = date.strftime('%Y-%m-%d')
            document_id = f"{labeled_to_original_mapping.get(sensor_id, sensor_id)}_{date_str}"

            data = {
                "_id": document_id,
                "sensor_id": labeled_to_original_mapping.get(sensor_id, sensor_id),
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

            data_dict = {key: float(value) if isinstance(value, (float, np.integer, float, np.floating)) else value
                         for key, value in data.items()}

            # Insert data into MongoDB
            collection.insert_one(data_dict)

        client.close()
        return

    except Exception as e:
        print(e)
@ensure_annotations
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
    # df['lag1'] = (df.index - pd.Timedelta('1 hour')).map(target_map)
    # df['lag2'] = (df.index - pd.Timedelta('12 hours')).map(target_map)
    # df['lag3'] = (df.index - pd.Timedelta('24 hours')).map(target_map)

    # '''will consider this in future'''
    df['lag1'] = (df.index - pd.Timedelta('30 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('60 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('90 days')).map(target_map)

    return df


@ensure_annotations
def rolling_statistics(self, data):
    MA = data.rolling(window=24).mean()
    MSTD = data.rolling(window=24).std()
    plt.figure(figsize=(15, 5))
    orig = plt.plot(data, color='black', label='Original')
    mean = plt.plot(MA, color='red', label='MA')
    std = plt.plot(MSTD, color='yellow', label='MSTD')
    plt.legend(loc='best')
    plt.title("Rolling Mean and standard Deviation")
    plt.show()


@ensure_annotations
def adfuller_test(self, data, sensor_id):
    print("Result od adifuller test:")
    # dftest = adfuller(data, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Stat', 'p-value', 'lags used', 'np of observation used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Values(%s)' % key] = value
    # print(f"The sensor id {sensor_id}-{dfoutput}")


@ensure_annotations
def store_actual_data(data):
    # print(data[['Clock', 'Kwh', 'sensor']])
    # return

    try:
        logger.info("calling DB configuration")
        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection3")

        MONGO_URL = f"mongodb://{host}:{port}"

        labeled_to_original_mapping = {
            0: "5f718c439c7a78.65267835",
            1: "62a9920f75c931.62399458",
        }

        client = MongoClient(MONGO_URL)
        db1 = client[db]
        collection = db1[collection_name]

        # Group data by date
        grouped_data = data.groupby(data['Clock'].dt.date)

        for date, group in grouped_data:
            date_str = date.strftime('%Y-%m-%d')
            document_id = f"{labeled_to_original_mapping.get(group['sensor'].iloc[0], group['sensor'].iloc[0])}_{date_str}"

            document = {
                "_id": document_id,
                "sensor_id": labeled_to_original_mapping.get(group['sensor'].iloc[0], group['sensor'].iloc[0]),
                "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "millisecond": int(datetime.now().timestamp() * 1000),
                "data": {}
            }

            # Populate the 'data' dictionary with hourly predictions
            for _, row in group.iterrows():
                actuals = round(float(row['Kwh']), 4)
                hour = row['Clock'].hour
                data_key = str(hour)
                data_value = {
                    "act_kwh": actuals
                }
                document["data"][data_key] = data_value

            # Insert data into MongoDB
            collection.insert_one(document)

        client.close()

    except Exception as e:
        print(e)