import os
import traceback

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.mlProject import logger
from src.mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            data_files = [file for file in os.listdir(self.config.train_data_path)]
            models_dict = {}

            for i in range(0, int(len(data_files) / 2)):
                X_train = pd.read_csv(os.path.join(self.config.train_data_path, f"train_data_sensor_{i}.csv"))
                y_train = pd.read_csv(os.path.join(self.config.train_data_path, f"test_data_sensor_{i}.csv"))

                X_train.set_index(['Clock'], inplace=True, drop=True)
                y_train.set_index(['Clock'], inplace=True, drop=True)

                # Train an XGBoost model on the sensor's data
                xgb_model = XGBRegressor()
                xgb_model.fit(X_train, y_train)

                # Calculate and print model evaluation metrics for this sensor
                # train_score = xgb_model.score(X_train, y_train)
                # print(f"Train Score for sensor {data_files}: {train_score}")

                # Perform hyperparameter tuning using RandomizedSearchCV
                best_params = {
                    'n_estimators': self.config.n_estimators,
                    'max_depth': self.config.max_depth,
                    'learning_rate': self.config.learning_rate,
                    'subsample': self.config.subsample,
                    'colsample_bytree': self.config.colsample_bytree
                }

                # Train the model with the best parameters
                best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'],
                                              max_depth=best_params['max_depth'],
                                              learning_rate=best_params['learning_rate'],
                                              subsample=best_params['subsample'],
                                              colsample_bytree=best_params['colsample_bytree'],
                                              reg_alpha=0.01,
                                              reg_lambda=0.01,
                                              min_child_weight=1,
                                              objective= 'reg:squarederror')
                best_xgb_model.fit(X_train, y_train)
                models_dict[i] = best_xgb_model
            # Save the dictionary of models as a single .joblib file
            joblib.dump(models_dict, os.path.join(self.config.root_dir, self.config.model_name))

        except Exception as e:
            traceback.print_exc()
            logger.info(f"Error in Model Trainer: {e}")
