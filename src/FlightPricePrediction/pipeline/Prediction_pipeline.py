import os
import sys
import pandas as pd

from src.FlightPricePrediction.exception import customexception
from src.FlightPricePrediction.logger import logging
from src.FlightPricePrediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            base_path = os.getcwd()

            # 🔥 TRY BOTH PATHS (safe for your repo structure)
            preprocessor_path = os.path.join(base_path, "artifacts", "preprocessor.pkl")
            model_path = os.path.join(base_path, "artifacts", "model.pkl")

            # fallback (if inside Notebook_Experiments)
            if not os.path.exists(preprocessor_path):
                preprocessor_path = os.path.join(base_path, "Notebook_Experiments", "artifacts", "preprocessor.pkl")

            if not os.path.exists(model_path):
                model_path = os.path.join(base_path, "Notebook_Experiments", "artifacts", "model.pkl")

            # 🔍 DEBUG
            print("📂 Preprocessor path:", preprocessor_path)
            print("📂 Model path:", model_path)
            print("Exists:", os.path.exists(preprocessor_path), os.path.exists(model_path))

            # load objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            # transform
            data_scaled = preprocessor.transform(features)

            # predict
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise customexception(e, sys)


class CustomData:
    def __init__(self,
                 airline: str,
                 source_city: str,
                 departure_time: str,
                 stops: str,
                 arrival_time: str,
                 destination_city: str,
                 classs: str,
                 duration: float,
                 days_left: int):
        
        self.airline = airline
        self.source_city = source_city
        self.departure_time = departure_time
        self.stops = stops
        self.arrival_time = arrival_time
        self.destination_city = destination_city
        self.classs = classs
        self.duration = duration
        self.days_left = days_left
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'airline': [self.airline],
                'source_city': [self.source_city],
                'departure_time': [self.departure_time],
                'stops': [self.stops],
                'arrival_time': [self.arrival_time],
                'destination_city': [self.destination_city],
                'class': [self.classs],   # 🔥 FIXED (IMPORTANT)
                'duration': [self.duration],
                'days_left': [self.days_left]
            }

            df = pd.DataFrame(custom_data_input_dict)

            logging.info('Dataframe Created Successfully')
            return df

        except Exception as e:
            logging.info('Exception Occured in CustomData')
            raise customexception(e, sys)