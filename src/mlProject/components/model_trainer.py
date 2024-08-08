import os
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject.utils.common import initialize_mongodb, data_fetching


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            client, collection = initialize_mongodb("transformed_data")
            models_dict = {}
            count = collection.count_documents({})
            print("no. of id's", count)

            for i in range(count):
                data = data_fetching(collection, i)
                if not data.empty:
                    # columns_to_drop = ['site_id', 'sensor_id']
                    # data.drop(columns=columns_to_drop, inplace = True)
                    data.set_index('Clock', inplace=True)
                    # print(data.info())
                    # return
                    #
                    # vif_data = pd.DataFrame()
                    # vif_data["feature"] = data.columns
                    # vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
                    # print(vif_data)
                    # return
                    # print(data.columns)

                    FEATURES = ['relative_humidity_2m', 'apparent_temperature', 'precipitation', 'wind_speed_10m',
                                'holiday', 'lag1', 'lag2', 'lag3', 'dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
                    TARGET = 'Kwh'

                    X = data[FEATURES]
                    y = data[TARGET]
                    total_length = len(X)

                    if total_length < 50:
                        print(f"Not enough data for sensor {i}. Skipping...")
                        continue

                    # Calculate dynamic split sizes
                    train_size = int(0.7 * total_length)
                    test_size = total_length - train_size
                    gap = 24

                    if train_size <= 0 or test_size <= 0 or train_size + test_size > total_length:
                        print(f"Data is not sufficient for proper splitting for sensor {i}. Skipping...")
                        continue

                    # Create fixed train-test split
                    # X_train = X[:train_size]
                    # y_train = y[:train_size]
                    # X_test = X[train_size:]
                    # y_test = y[train_size:]

                    # Create TimeSeriesSplit object for training data
                    # tscv = TimeSeriesSplit(n_splits=5)
                    # df = data.sort_index()
                    #
                    # preds = []
                    # scores = []
                    # # Perform Time Series Cross-Validation
                    # for train_idx, val_idx in tscv.split(df):
                    #     train_data = df.iloc[train_idx]
                    #     test_data = df.iloc[val_idx]
                    #     X_train = train_data[FEATURES]
                    #     y_train = train_data[TARGET]
                    #     X_test = test_data[FEATURES]
                    #     y_test = test_data[TARGET]
                    #     # Train an XGBoost model on the sensor's data
                    #     xgb_model = XGBRegressor()
                    #     xgb_model.fit(X_train, y_train)
                    #
                    #     # Calculate and print model evaluation metrics for this sensor
                    #     train_score = xgb_model.score(X_train, y_train)
                    #     val_score = xgb_model.score(X_test, y_test)
                    #     # print(f"Train Score for sensor {i}: {train_score}")
                    #     # print(f"Validation Score for sensor {i}: {val_score}")
                    #
                    #     # Perform hyperparameter tuning using RandomizedSearchCV
                    #     param_grid = {
                    #         'n_estimators': self.config.n_estimators,
                    #         'max_depth': self.config.max_depth,
                    #         'learning_rate': self.config.learning_rate,
                    #         'subsample': self.config.subsample,
                    #         'colsample_bytree': self.config.colsample_bytree,
                    #         'reg_alpha': self.config.reg_alpha,
                    #         'reg_lambda': [0.01, 0.1, 1]
                    #     }
                    #
                    #     random_search = RandomizedSearchCV(xgb_model,
                    #                                        param_distributions=param_grid,
                    #                                        n_iter=10,
                    #                                        scoring='neg_mean_squared_error',
                    #                                        cv=tscv,
                    #                                        random_state=100)
                    #
                    #     # Fit the RandomizedSearchCV to the data
                    #     random_search.fit(X_train, y_train)
                    #
                    #     # Get the best parameters
                    #     best_params = random_search.best_params_
                    #     # print(f"Best Parameters for sensor {i}: {best_params}")
                    #
                    #     # Train the model with the best parameters
                    #     best_xgb_model = XGBRegressor(base_score=0.5, booster='gbtree',
                    #                                   n_estimators=1000,
                    #                                   early_stopping_rounds=50,
                    #                                   objective='reg:squarederror',
                    #                                   max_depth=3,
                    #                                   learning_rate=0.01)
                    #     best_xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
                    #     y_pred = best_xgb_model.predict(X_test)
                    #     preds.append(y_pred)
                    #     score = np.sqrt(mean_squared_error(y_test, y_pred))
                    #     scores.append(score)
                    #     print(scores)
                        # Evaluate the model on the test set
                        # test_score = best_xgb_model.score(X_test, y_test)
                        # print(f"Test Score for sensor {i}: {test_score}")
                        # models_dict[i] = best_xgb_model
                    # '''==================================================Making model on full data========================================================================'''

                    xgb_model = XGBRegressor()
                    xgb_model.fit(X, y)
                    #
                    # Calculate and print model evaluation metrics for this sensor
                    train_score = xgb_model.score(X, y)
                    # val_score = xgb_model.score(X_test, y_test)
                    print(f"Train Score for sensor {i}: {train_score}")
                    # print(f"Validation Score for sensor {i}: {val_score}")

                    # Perform hyperparameter tuning using RandomizedSearchCV
                    param_grid = {
                        'n_estimators': self.config.n_estimators,
                        'max_depth': self.config.max_depth,
                        'learning_rate': self.config.learning_rate,
                        'subsample': self.config.subsample,
                        'colsample_bytree': self.config.colsample_bytree,
                        'reg_alpha': self.config.reg_alpha
                        # 'reg_lambda': [0.01, 0.1, 1]
                    }

                    random_search = RandomizedSearchCV(xgb_model,
                                                       param_distributions=param_grid,
                                                       n_iter=5,
                                                       scoring='neg_mean_squared_error',
                                                       cv=5,
                                                       random_state=100)

                    # Fit the RandomizedSearchCV to the data
                    # random_search.fit(X_train, y_train)
                    random_search.fit(X, y)

                    # Get the best parameters
                    best_params = random_search.best_params_
                    print(f"Best Parameters for sensor {i}: {best_params}")

                    # Train the model with the best parameters
                    best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                                  max_depth=best_params['max_depth'],
                                                  learning_rate=best_params['learning_rate'],
                                                  subsample=best_params['subsample'],
                                                  colsample_bytree=best_params['colsample_bytree'],
                                                  reg_alpha=best_params['reg_alpha'],
                                                  base_score=0.5,
                                                  booster='gbtree',
                                                  # reg_lambda=best_params['reg_lambda'],
                                                  objective='reg:squarederror')
                    best_xgb_model.fit(X, y)

                    # Evaluate the model on the test set
                    test_score = best_xgb_model.score(X, y)
                    print(f"Test Score for sensor {i}: {test_score}")

                    models_dict[i] = best_xgb_model

            # Save the dictionary of models as a single .joblib file
            joblib.dump(models_dict, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")

        finally:
            client.close()
            print("db connection closed")
