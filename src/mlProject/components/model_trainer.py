import os
import traceback

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import  TimeSeriesSplit
from src.mlProject.utils.common import initialize_mongodb, data_fetching


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        # self.test  =  test_data

    def train(self):
        try:
            
            client, collection = initialize_mongodb("train")
            models_dict = {}
            count = collection.count_documents({})
            print("no. of id's",count)

            for i in range(count):

                train_data , name = data_fetching(collection,i)
                train_data.set_index(['Clock'], inplace=True, drop=True)
                
                FEATURES = ['labeled_id', 'holiday', 'humidity', 'rain','cloud_cover', 'wind_speed', 'temp_diff',
                            'lag1', 'lag2', 'lag3','lag4', 'lag5', 'lag6', 'day', 'hour', 'month',
                            'dayofweek', 'quarter','dayofyear', 'weekofyear', 'year']
                TARGET = ['Kwh']

                X_train = train_data[FEATURES]
                y_train = train_data[TARGET]

                # Train an XGBoost model on the sensor's data
                xgb_model = XGBRegressor()
                xgb_model.fit(X_train, y_train)

                # Calculate and print model evaluation metrics for this sensor
                train_score = xgb_model.score(X_train, y_train)
                print(f"Train Score for sensor {i}: {train_score}")

                # Perform hyperparameter tuning using RandomizedSearchCV
                param_grid = {
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'subsample': self.config.subsample
                }

                random_search = RandomizedSearchCV(xgb_model,
                                                   param_distributions=param_grid,
                                                   n_iter=10,
                                                   scoring='neg_mean_squared_error',
                                                   cv=5,
                                                   # verbose=1,
                                                   n_jobs=-1,
                                                   random_state=42)

                # Fit the RandomizedSearchCV to the data
                random_search.fit(X_train, y_train)

                # Get the best parameters
                best_params = random_search.best_params_
                print(f"Best Parameters for sensor {i}: {best_params}")

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'],
                                              subsample=best_params['subsample'],
                                              reg_alpha=0.01,
                                              reg_lambda=0.01)
                best_xgb_model.fit(X_train, y_train)

                models_dict[i] = best_xgb_model
                # print(models_dict)
            # Save the dictionary of models as a single .joblib file
            joblib.dump(models_dict, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")

        finally:
            client.close()
            print("db connection closed")

