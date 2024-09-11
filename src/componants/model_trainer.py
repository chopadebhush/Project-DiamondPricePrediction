import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info(
                "Splitting dependant & indepenadant variable from train and test data")
            # Lets devide train array and test array for model training
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            # Initialized all required model here
            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor()
            }
            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)
            print(
                "\n==================================================================================")
            logging.info(f'Model_report":{model_report}')

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list
                                                        (model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            # Print model name and r2 score
            print(
                f'Best model found, Model Name:{best_model_name},R2_score:{best_model_score}')
            print(
                '\n=============================================================================')
            logging.info(
                f'Best model found, Model Name:{best_model_name},R2_score:{best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e, sys)
