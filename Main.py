#!/usr/bin/env python3
import os
import logging
import argparse
import sys
from pathlib import Path
import traceback
from datetime import datetime

# Import your modules
from Data.DataMain import DataModule
from Data.DataStorage import StorageModule
from MLModule.Model1 import RecommendationModule

class PipelineManager:
    """Manager class to handle the recommendation pipeline process"""
    
    def __init__(self, input_file, storage_dir, log_level=logging.INFO, customer_id=None):
        """Initialize the pipeline manager with configuration"""
        self.input_file = input_file
        self.storage_dir = storage_dir
        self.customer_id = customer_id
        self.setup_logging(log_level)
        self.create_directories()
        
    def setup_logging(self, log_level):
        """Set up logging with appropriate format and level"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("pipeline")
        self.logger.info(f"Logging initialized at level {log_level}")
        
    def create_directories(self):
        """Ensure all necessary directories exist"""
        for directory in ["data", self.storage_dir]:
            path = Path(directory)
            path.mkdir(exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {path}")
            
    def process_data(self):
        """Step 1: Process input data"""
        self.logger.info("Step 1: Processing data")
        try:
            self.data_module = DataModule()
            self.data_module.load_csv(self.input_file)
            self.logger.info(f"Loaded data from {self.input_file}")
            
            self.data_module.clean_data()
            self.logger.info("Data cleaned successfully")
            
            self.recommendation_data = self.data_module.prepare_data_for_recommendations()
            self.logger.info("Data prepared for recommendations")
            return True
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def save_processed_data(self):
        """Step 2: Save processed data"""
        self.logger.info("Step 2: Saving processed data")
        try:
            self.storage_module = StorageModule(storage_dir=self.storage_dir)
            self.version = self.storage_module.save_processed_data(self.recommendation_data)
            self.logger.info(f"Data saved as version: {self.version}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def load_processed_data(self):
        """Step 3: Load processed data"""
        self.logger.info("Step 3: Loading processed data")
        try:
            self.loaded_data = self.storage_module.load_processed_data()
            self.logger.info(f"Data loaded successfully with keys: {list(self.loaded_data.keys())}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def initialize_recommendation_module(self):
        """Step 4: Initialize recommendation module"""
        self.logger.info("Step 4: Initializing recommendation module")
        try:
            self.recommendation_module = RecommendationModule()
            self.logger.info("Recommendation module initialized")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing recommendation module: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def load_data_into_recommendation_module(self):
        """Step 5: Load data into recommendation module"""
        self.logger.info("Step 5: Loading data into recommendation module")
        try:
            self.recommendation_module.load_data(self.loaded_data)
            self.logger.info("Data loaded into recommendation module")
            return True
        except Exception as e:
            self.logger.error(f"Error loading data into recommendation module: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def get_recommendations(self, customer_id=None):
        """Step 6: Generate recommendations for a customer"""
        customer = customer_id or self.customer_id
        if not customer:
            self.logger.warning("No customer ID provided for recommendations")
            return None
            
        self.logger.info(f"Step 6: Getting recommendations for customer {customer}")
        try:
            recommendations = self.recommendation_module.recommend_for_customer(customer)
            self.logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def run_pipeline(self):
        """Execute the full pipeline"""
        self.logger.info("Starting recommendation pipeline")
        
        # Run all steps sequentially, stopping if any step fails
        if not self.process_data():
            self.logger.error("Pipeline failed at data processing step")
            return False
            
        if not self.save_processed_data():
            self.logger.error("Pipeline failed at data saving step")
            return False
            
        if not self.load_processed_data():
            self.logger.error("Pipeline failed at data loading step")
            return False
            
        if not self.initialize_recommendation_module():
            self.logger.error("Pipeline failed at recommendation module initialization")
            return False
            
        if not self.load_data_into_recommendation_module():
            self.logger.error("Pipeline failed at loading data into recommendation module")
            return False
            
        if self.customer_id:
            recommendations = self.get_recommendations()
            if recommendations:
                self.logger.info(f"Recommendations for {self.customer_id}: {recommendations}")
            else:
                self.logger.warning("Failed to generate recommendations")
        else:
            self.logger.info("No customer ID provided, skipping recommendation generation")
            
        self.logger.info("Pipeline completed successfully")
        return True


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Recommendation Pipeline")
    parser.add_argument("--input", "-i", default="data/transaction_data.csv", 
                        help="Input CSV file path")
    parser.add_argument("--storage", "-s", default="data_storage", 
                        help="Storage directory for processed data")
    parser.add_argument("--customer", "-c", 
                        help="Customer ID for recommendations")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Initialize and run pipeline
    pipeline = PipelineManager(
        input_file=args.input,
        storage_dir=args.storage,
        log_level=log_level,
        customer_id=args.customer
    )
    
    success = pipeline.run_pipeline()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())