import os
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
from Data.DataMain import *
from Data.DataStorage import *
from MLModule.Model1 import *

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("data_storage", exist_ok=True)

print("Step 1: Process data")
data_module = DataModule()
data_module.load_csv("data/transaction_data.csv")
data_module.clean_data()
recommendation_data = data_module.prepare_data_for_recommendations()

print("Step 2: Save processed data")
storage_module = StorageModule(storage_dir="data_storage")
version = storage_module.save_processed_data(recommendation_data)
print(f"Data saved as version: {version}")

print("Step 3: Load processed data")
loaded_data = storage_module.load_processed_data()
print("Data keys:", list(loaded_data.keys()))

print("Step 4: Initialize recommendation module")
recommendation_module = RecommendationModule()
print("Recommendation module initialized")

print("Step 5: Load data into recommendation module")
recommendation_module.load_data(loaded_data)
print("Data loaded into recommendation module")

print("Step 6: Try to get recommendations")
try:
    recommendations = recommendation_module.recommend_for_customer("C0407")
    print("Recommendations:", recommendations)
except Exception as e:
    print(f"Error getting recommendations: {str(e)}")