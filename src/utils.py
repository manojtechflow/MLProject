import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import dill

from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error(CustomException(e,sys))
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name, model_instance in models.items():

            model_instance.fit(X_train, y_train)

            y_test_pred = model_instance.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report
    except Exception as e:
        logging.error(CustomException(e, sys))
        raise CustomException(e, sys)