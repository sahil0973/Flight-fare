import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================
# SAVE OBJECT
# =========================
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        print("💾 Saving file:", file_path)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        print("✅ File saved successfully")

    except Exception as e:
        raise customexception(e, sys)


# =========================
# LOAD OBJECT (IMPORTANT)
# =========================
def load_object(file_path):
    try:
        # 🔍 check file exists
        if not os.path.exists(file_path):
            raise Exception(f"❌ File not found: {file_path}")

        # 🔍 check file size
        file_size = os.path.getsize(file_path)

        print("📂 Loading file:", file_path)
        print("📏 File size:", file_size, "bytes")

        if file_size < 100:   # too small → invalid pickle
            raise Exception(f"❌ File is too small / corrupt: {file_path}")

        # 🔥 load pickle
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)

        print("✅ File loaded successfully")

        return obj

    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e, sys)


# =========================
# MODEL EVALUATION
# =========================
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]

            # train
            model.fit(X_train, y_train)

            # predict
            y_test_pred = model.predict(X_test)

            # score
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e, sys)