import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from src.componants.data_ingestion import DataIngestion
from src.componants.data_transformation import DataTransformation
from src.componants.model_trainer import ModelTrainer


if __name__ == "__main__":
    # Data ingestion object
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    # Data transformation object
    data_transformation = DataTransformation()
    train_arr, test_arr, obj_path = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path)

    # Model training object
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr)
