import os
import logging
import argparse
import pandas as pd
from datetime import datetime

from Data.DataMain import *
from Data.DataStorage import *
from MLModule.Model1 import *

class RecommendationSystem:
    """
    Main module that orchestrates the entire recommendation system workflow.
    Implements a strategy pattern to coordinate data processing, storage, and recommendation generation.
    """
    
    def __init__(self, data_dir="data", storage_dir="data_storage", log_level=logging.INFO):
        """
        Initialize the recommendation system
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory where raw data files are stored (default: "data")
        storage_dir : str, optional
            Directory where processed data will be stored (default: "data_storage")
        log_level : int, optional
            Logging level (default: logging.INFO)
        """
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("recommendation_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RecommendationSystem')
        
        # Initialize modules
        self.data_module = DataModule()
        self.storage_module = StorageModule(storage_dir=storage_dir)
        self.recommendation_module = RecommendationModule()
        
        # Store configurations
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        
        self.logger.info("Recommendation System initialized")
    
    def process_data(self, file_path, encoding='utf-8', save_version=None):
        """
        Process new transaction data
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing transaction data
        encoding : str, optional
            Character encoding to use when reading the file (default: 'utf-8')
        save_version : str, optional
            Version identifier for saving the processed data (default: timestamp)
            
        Returns:
        --------
        dict
            Summary of the processed data
        """
        self.logger.info(f"Processing data from {file_path}")
        
        try:
            # Load and process data
            self.data_module.load_csv(file_path, encoding=encoding)
            self.data_module.clean_data()
            recommendation_data = self.data_module.prepare_data_for_recommendations()
            
            # Save processed data
            if save_version is None:
                save_version = datetime.now().strftime("%Y%m%d_%H%M%S")
                
            self.storage_module.save_processed_data(recommendation_data, version=save_version)
            
            # Get data summary
            summary = self.data_module.get_data_summary()
            self.logger.info(f"Data processing completed for version {save_version}")
            
            return {
                'version': save_version,
                'summary': summary,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def load_model(self, version=None):
        """
        Load processed data for making recommendations
        
        Parameters:
        -----------
        version : str, optional
            Version identifier to load (default: latest version)
            
        Returns:
        --------
        bool
            True if loading was successful
        """
        try:
            # Load data from storage
            recommendation_data = self.storage_module.load_processed_data(version=version)
            
            # Initialize recommendation module with loaded data
            self.recommendation_module.load_data(recommendation_data)
            
            version_info = self.storage_module.get_version_info(version)
            self.logger.info(f"Successfully loaded model version {version_info.get('version', 'unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_recommendations(self, customer_id=None, product_id=None, n=5, method='hybrid'):
        """
        Generate recommendations based on customer ID or product ID
        
        Parameters:
        -----------
        customer_id : str, optional
            ID of the customer to generate recommendations for
        product_id : str, optional
            ID of the product to find similar items for
        n : int, optional
            Number of recommendations to generate (default: 5)
        method : str, optional
            Recommendation method to use (default: 'hybrid')
            
        Returns:
        --------
        dict
            Dictionary containing recommendations
        """
        try:
            if customer_id is not None:
                # Generate customer-based recommendations
                recommendations = self.recommendation_module.recommend_for_customer(
                    customer_id=customer_id, 
                    n=n, 
                    include_purchased=False
                )
                
                return {
                    'customer_id': customer_id,
                    'recommendations': recommendations,
                    'type': 'customer_based',
                    'status': 'success'
                }
                
            elif product_id is not None:
                # Generate product-based recommendations
                recommendations = self.recommendation_module.get_similar_products(
                    product_id=product_id, 
                    n=n, 
                    method=method
                )
                
                return {
                    'product_id': product_id,
                    'recommendations': recommendations,
                    'type': 'product_based',
                    'method': method,
                    'status': 'success'
                }
                
            else:
                return {
                    'status': 'error',
                    'error': 'Either customer_id or product_id must be provided'
                }
                
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def export_analytics(self, output_dir="analytics"):
        """
        Export analytics data for reporting and visualization
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory where analytics files will be saved (default: "analytics")
            
        Returns:
        --------
        dict
            Dictionary containing paths to exported files
        """
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Exporting analytics to {output_dir}")
        
        try:
            # Export customer data
            customer_path = os.path.join(output_dir, "customer_analysis.csv")
            self.storage_module.export_customer_data(customer_path)
            
            # Export product data
            product_path = os.path.join(output_dir, "product_analysis.csv")
            self.storage_module.export_product_data(product_path)
            
            # Generate additional analytics files
            version_info = self.storage_module.get_version_info()
            versions_list = self.storage_module.list_available_versions()
            
            # Save model version info
            version_path = os.path.join(output_dir, "model_versions.csv")
            pd.DataFrame({
                'version': versions_list,
                'current': [v == version_info.get('version', '') for v in versions_list]
            }).to_csv(version_path, index=False)
            
            return {
                'customer_analysis': customer_path,
                'product_analysis': product_path,
                'model_versions': version_path,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting analytics: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_batch_recommendations(self, batch_file, output_file):
        """
        Generate recommendations for a batch of customers or products
        
        Parameters:
        -----------
        batch_file : str
            Path to CSV file containing IDs to generate recommendations for
            (should have either 'customer_id' or 'product_id' column)
        output_file : str
            Path where recommendations will be saved
            
        Returns:
        --------
        dict
            Summary of batch processing
        """
        self.logger.info(f"Running batch recommendations from {batch_file}")
        
        try:
            # Load batch file
            batch_df = pd.read_csv(batch_file)
            
            # Determine batch type
            is_customer_batch = 'customer_id' in batch_df.columns
            is_product_batch = 'product_id' in batch_df.columns
            
            if not (is_customer_batch or is_product_batch):
                raise ValueError("Batch file must contain either 'customer_id' or 'product_id' column")
            
            # Initialize results
            recommendations = []
            processed_count = 0
            error_count = 0
            
            # Process each item
            if is_customer_batch:
                for customer_id in batch_df['customer_id'].unique():
                    try:
                        result = self.get_recommendations(customer_id=customer_id)
                        
                        if result['status'] == 'success':
                            # Add recommendations for this customer
                            for rank, product_id in enumerate(result['recommendations'], 1):
                                recommendations.append({
                                    'customer_id': customer_id,
                                    'product_id': product_id,
                                    'rank': rank,
                                    'recommendation_type': 'customer_based'
                                })
                            processed_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing customer {customer_id}: {str(e)}")
                        error_count += 1
            
            elif is_product_batch:
                for product_id in batch_df['product_id'].unique():
                    try:
                        result = self.get_recommendations(product_id=product_id)
                        
                        if result['status'] == 'success':
                            # Add recommendations for this product
                            for rank, similar_id in enumerate(result['recommendations'], 1):
                                recommendations.append({
                                    'product_id': product_id,
                                    'similar_product_id': similar_id,
                                    'rank': rank,
                                    'recommendation_type': 'product_based'
                                })
                            processed_count += 1
                        else:
                            error_count += 1
                            
                    except Exception as e:
                        self.logger.error(f"Error processing product {product_id}: {str(e)}")
                        error_count += 1
            
            # Save results
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_csv(output_file, index=False)
            
            return {
                'total_items': len(batch_df),
                'processed_count': processed_count,
                'error_count': error_count,
                'output_file': output_file,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_model_stats(self):
        """
        Get statistics about the current model
        
        Returns:
        --------
        dict
            Dictionary containing model statistics
        """
        try:
            # Get version info
            version_info = self.storage_module.get_version_info()
            
            # Get available versions
            versions = self.storage_module.list_available_versions()
            
            # Get data summary if available
            data_summary = None
            if hasattr(self.data_module, 'get_data_summary'):
                try:
                    data_summary = self.data_module.get_data_summary()
                except:
                    pass
            
            return {
                'current_version': version_info.get('version'),
                'created_at': version_info.get('created_at'),
                'available_versions': versions,
                'data_summary': data_summary,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model stats: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

def main():
    """
    Main function to run the recommendation system from command line
    """
    parser = argparse.ArgumentParser(description="Product Recommendation System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process data command
    process_parser = subparsers.add_parser("process", help="Process transaction data")
    process_parser.add_argument("--file", required=True, help="Path to transaction data CSV file")
    process_parser.add_argument("--encoding", default="utf-8", help="File encoding")
    process_parser.add_argument("--version", help="Version identifier for the processed data")
    
    # Load model command
    load_parser = subparsers.add_parser("load", help="Load a model version")
    load_parser.add_argument("--version", help="Version to load (default: latest)")
    
    # Get recommendations command
    recommend_parser = subparsers.add_parser("recommend", help="Generate recommendations")
    recommend_parser.add_argument("--customer", help="Customer ID to generate recommendations for")
    recommend_parser.add_argument("--product", help="Product ID to find similar items for")
    recommend_parser.add_argument("--count", type=int, default=5, help="Number of recommendations")
    recommend_parser.add_argument("--method", default="hybrid", choices=["cooccurrence", "content", "hybrid"], 
                                 help="Recommendation method")
    
    # Export analytics command
    export_parser = subparsers.add_parser("export", help="Export analytics data")
    export_parser.add_argument("--output", default="analytics", help="Output directory")
    
    # Batch recommendations command
    batch_parser = subparsers.add_parser("batch", help="Generate recommendations for a batch of items")
    batch_parser.add_argument("--input", required=True, help="Input batch CSV file")
    batch_parser.add_argument("--output", required=True, help="Output recommendations CSV file")
    
    # Model stats command
    stats_parser = subparsers.add_parser("stats", help="Show model statistics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize recommendation system
    system = RecommendationSystem()
    
    # Execute command
    if args.command == "process":
        result = system.process_data(args.file, encoding=args.encoding, save_version=args.version)
        print(f"Data processing {'successful' if result['status'] == 'success' else 'failed'}")
        if result['status'] == 'success':
            print(f"Version: {result['version']}")
            print("Summary:")
            for key, value in result['summary'].items():
                print(f"  {key}: {value}")
    
    elif args.command == "load":
        success = system.load_model(version=args.version)
        print(f"Model loading {'successful' if success else 'failed'}")
    
    elif args.command == "recommend":
        if args.customer:
            result = system.get_recommendations(customer_id=args.customer, n=args.count)
            if result['status'] == 'success':
                print(f"Recommendations for customer {args.customer}:")
                for i, product_id in enumerate(result['recommendations'], 1):
                    print(f"  {i}. {product_id}")
        elif args.product:
            result = system.get_recommendations(product_id=args.product, n=args.count, method=args.method)
            if result['status'] == 'success':
                print(f"Similar products to {args.product}:")
                for i, product_id in enumerate(result['recommendations'], 1):
                    print(f"  {i}. {product_id}")
        else:
            print("Error: Either --customer or --product must be specified")
    
    elif args.command == "export":
        result = system.export_analytics(output_dir=args.output)
        if result['status'] == 'success':
            print(f"Analytics exported to {args.output}")
            for key, path in result.items():
                if key != 'status':
                    print(f"  {key}: {path}")
    
    elif args.command == "batch":
        result = system.run_batch_recommendations(args.input, args.output)
        if result['status'] == 'success':
            print(f"Batch processing completed:")
            print(f"  Total items: {result['total_items']}")
            print(f"  Processed: {result['processed_count']}")
            print(f"  Errors: {result['error_count']}")
            print(f"  Results saved to: {result['output_file']}")
    
    elif args.command == "stats":
        result = system.get_model_stats()
        if result['status'] == 'success':
            print("Model Statistics:")
            print(f"  Current version: {result['current_version']}")
            print(f"  Created at: {result['created_at']}")
            print(f"  Available versions: {', '.join(result['available_versions'])}")
            if result['data_summary']:
                print("  Data Summary:")
                for key, value in result['data_summary'].items():
                    print(f"    {key}: {value}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()