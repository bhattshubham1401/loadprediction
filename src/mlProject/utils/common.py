'''functionality that we are use in our code'''

import json
import os
from datetime import date as datetime_date, timedelta, datetime
import holidays
from pathlib import Path
from typing import Any

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
from pymongo import MongoClient
from sklearn.model_selection import train_test_split , TimeSeriesSplit

# from src.mlProject.components.data_transformation import sensorDecode

from statsmodels.tsa.stattools import adfuller

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
        lst = [
            '5f718b613291c7.03696209',
            '5f718c439c7a78.65267835',
            '614366bce31a86.78825897',
            '6148740eea9db0.29702291',
            '625fb44c5fb514.98107900',
            '625fb9e020ff31.33961816',
            '6260fd4351f892.69790282',
            '627cd4815f2381.31981050',
            '629094ee5fdff4.43505210',
            '62aad7f5c65185.80723547',
            '62b15dfee341d1.73837476',
            '62b595eabd9df4.71374208',
            '6349368c306542.16235883',
            '634e7c43038801.39310596',
            '6399a18b1488b8.07706749',
            '63a4195534d625.00718490',
            '63a4272631f153.67811394',
            '63aa9161b9e7e1.16208626',
            '63ca403ccd66f3.47133508',
            '62a9920f75c931.62399458'
            ]
    
        l1 = []
        toDate = datetime(2023, 11, 18, 23, 59, 59)

        for sensor in lst:
            url = "https://multipoint.myxenius.com/Sensor_newHelper/getDataApi"

            params = {
                'sql': "select raw_data, sensor_id from dlms_load_profile where sensor_id='{}' and read_time<='{}' order by read_time".format(
                    sensor, toDate),
                'type': 'query'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            l1.append(data['resource'])

        columns = ['sensor', 'Clock', 'R_Voltage', 'Y_Voltage', 'B_Voltage', 'R_Current', 'Y_Current',
                   'B_Current', 'A', 'BlockEnergy-WhExp', 'B', 'C', 'D', 'BlockEnergy-VAhExp',
                   'Kwh', 'BlockEnergy-VArhQ1', 'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp']

        datalist = [(entry['sensor_id'], entry['raw_data']) for i in range(len(l1)) for entry in l1[i]]

        df = pd.DataFrame([row[0].split(',') + row[1].split(',') for row in datalist], columns=columns)

        df = df.drop([
            'BlockEnergy-WhExp', 'A', 'B', 'C', 'D', 'BlockEnergy-VAhExp', 'BlockEnergy-VAhExp', 'BlockEnergy-VArhQ1',
            'BlockEnergy-VArhQ4', 'BlockEnergy-VAhImp'], axis=1)
        pd.set_option('display.max_columns', None)

        df['Clock'] = pd.to_datetime(df['Clock'])
        df['Kwh'] = df['Kwh'].astype(float)
        df['R_Voltage'] = df['R_Voltage'].astype(float)
        df['Y_Voltage'] = df['Y_Voltage'].astype(float)
        df['B_Voltage'] = df['B_Voltage'].astype(float)
        df['R_Current'] = df['R_Current'].astype(float)
        df['Y_Current'] = df['Y_Current'].astype(float)
        df['B_Current'] = df['B_Current'].astype(float)
        # print(df.tail())
        return df

    except Exception as e:
        logger.info(f"Error occurs =========== {e}")


@ensure_annotations
def load_file():
    file = os.getenv("filename")
    return file


@ensure_annotations
# def plotData(df1):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df1['KWh'], df1['cumm_PF'], label='Actual')
#     plt.xlabel('KWh ')
#     plt.ylabel('cumm_PF')
#     plt.legend()
#     plt.show()
#     return

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

        # Example mapping:
        # labeled_to_original_mapping = {
        #     "0": "5f718c439c7a78.65267835",
        #     "1": "62a9920f75c931.62399458",
        # }

        db = os.getenv("db")
        host = os.getenv("host")
        port = os.getenv("port")
        collection_name = os.getenv("collection2")

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
    # print(df)
    target_map = df['Kwh'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('30 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('60 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('90 days')).map(target_map)

    df['lag4'] = (df.index - pd.Timedelta('1 hour')).map(target_map)
    df['lag5'] = (df.index - pd.Timedelta('12 hours')).map(target_map)
    df['lag6'] = (df.index - pd.Timedelta('24 hours')).map(target_map)

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
def data_from_weather_api():
    ''' weather data'''
    logger.info("weather data fetching")
    try:
        l1=[]
        value=[]
        start_date = "2022-11-18"
        end_date = "2023-11-18"
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude=28.58&longitude=77.33&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m&timezone=auto"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        l1.append(data)
        
        data1=l1[0]
        data_dict=data1['hourly']
        
        for i in range(len(data_dict['time'])):
            value.append({
                        "Clock":data_dict['time'][i],
                        "temp":data_dict['temperature_2m'][i],
                        "humidity":data_dict['relative_humidity_2m'][i],
                        "rain":data_dict['rain'][i],
                        "cloud_cover":data_dict['cloud_cover'][i],
                        "wind_speed":data_dict['wind_speed_10m'][i],
                        
                        })
        df=pd.DataFrame(value)
        df['Clock']=pd.to_datetime(df['Clock'])
        df.set_index("Clock",inplace=True, drop=True)
        df["temp_diff"]=df['temp']-df['temp'].shift(1)
        df.drop(['temp'],axis=1,inplace=True)
        df.fillna(value=0,inplace=True)
        # df.dropna(inplace=True)
        return df
    
    except Exception as e:
        print(e) 

@ensure_annotations
def holidays_list():
    logger.info("holidays list")
    try:
        start_date = datetime_date(2023, 1, 1)
        end_date = datetime_date(2023, 11, 18)
        holiday_list = []

        def is_holiday(single_date):
            year = single_date.year
            country_holidays = holidays.CountryHoliday('India', years=year)
            return single_date in country_holidays

        date_list = [
            (single_date, holidays.CountryHoliday('India', years=single_date.year)[single_date])
            for single_date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1))
            if is_holiday(single_date)]
        for date, name in date_list:
            # print(f"{date}: {name}")
            holiday_list.append(date)

        return holiday_list
    except Exception as e:
            print(e)

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
        collection_name3 = os.getenv("collection3")
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
        collection3 = db1[collection_name3]
       
        if type(collection_name)== list:
            return db1, client, collection1, collection2, collection3
        elif collection_name == "sensor":
            return client, collection1
        elif collection_name == "train":
            return client, collection2
        elif collection_name == "test":
            return client, collection3
    except Exception as e:
            print(e)  

@ensure_annotations
# def store_sensor_data_in_db(sensor_id, df,collection, db, sensorName)
def store_sensor_data_in_db(db, collection1, collection2, collection3, sensor_id,\
                             sensorName, dfresample, train_data, test_data):
    try:
        if sensor_id == 0 :
            logger.info("Data found")
            # db.collection1.drop()
            # db.collection2.drop()
            # db.collection3.drop()
            db.drop_collection(collection1.name)
            db.drop_collection(collection2.name)
            db.drop_collection(collection3.name)

        sensor_data1 = dfresample.to_dict(orient='records')
        id_data1 = { "sensor_id": sensor_id, "sensor_name": sensorName, "data": sensor_data1 }
        
        train_data1 = train_data.to_dict(orient='records')
        id_data2 = { "sensor_id": sensor_id, "sensor_name": sensorName, "data": train_data1 }
        
        test_data1 = test_data.to_dict(orient='records')
        id_data3 = { "sensor_id": sensor_id, "sensor_name": sensorName, "data": test_data1 }
        
        collection1.insert_one(id_data1)
        collection2.insert_one(id_data2)
        collection3.insert_one(id_data3)
        logger.info("Data stored")

    except Exception as e:
        print(e)    

@ensure_annotations
def data_fetching(collection, i):
    try:
        data_list = list(collection.find({"sensor_id":i}, {'_id': 0,"sensor_name" : 1, 'data': 1}))
        df = pd.DataFrame(data_list[0]['data'])
        sensor_name = data_list[0]['sensor_name']
        return df, sensor_name
    except Exception as e:
        print(e)    

@ensure_annotations
def uom():
    try:
        sensor_ids = [
            '5f718b613291c7.03696209','5f718c439c7a78.65267835',
            '614366bce31a86.78825897','6148740eea9db0.29702291','625fb44c5fb514.98107900',
            '625fb9e020ff31.33961816','6260fd4351f892.69790282','627cd4815f2381.31981050',
            '629094ee5fdff4.43505210','62aad7f5c65185.80723547','62b15dfee341d1.73837476',
            '62b595eabd9df4.71374208','6349368c306542.16235883','634e7c43038801.39310596',
            '6399a18b1488b8.07706749','63a4195534d625.00718490','63a4272631f153.67811394',
            '63aa9161b9e7e1.16208626','63ca403ccd66f3.47133508','62a9920f75c931.62399458'
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
        if id == 0 :
            db.drop_collection(collection_name)
            logger.info("deleted found test_data")

        data1 = df.to_dict(orient='records')
        id_data = { "sensor_id": id, "data": data1 }

        collection.insert_one(id_data)
    except Exception as e:
        print(e)

@ensure_annotations
def test_data_fetching(collection, i):
    try:
        data_list = list(collection.find({"sensor_id":i}, {'_id': 0,"sensor_name" : 1, 'data': 1}))
        df = pd.DataFrame(data_list[0]['data'])
        sensor_name = data_list[0]['sensor_name']
        return df, sensor_name
    except Exception as e:
        print(e)    