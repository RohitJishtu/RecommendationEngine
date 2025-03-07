import pickle
import json
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

class StorageModule:
    """
    Storage Module for persisting and retrieving processed data from the DataModule.
    Handles saving and loading data structures, matrices, and metadata to/from disk.
    """
    
    def __init__(self, storage_dir="data_storage"):
        """
        Initialize the storage module
        
        Parameters:
        -----------
        storage_dir : str, optional
            Directory path where data will be stored (default: "data_storage")
        """
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('StorageModule')
    
    def save_processed_data(self, recommendation_data, version=None):
        """
        Save all processed data from the DataModule
        
        Parameters:
        -----------
        recommendation_data : dict
            Dictionary containing all processed data
        version : str, optional
            Version identifier for the data (default: current timestamp)
            
        Returns:
        --------
        str
            Path to the directory where data was saved
        """
        # Generate version identifier if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create version directory
        version_dir = os.path.join(self.storage_dir, f"version_{version}")
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
            
        self.logger.info(f"Saving processed data to {version_dir}")
        
        # Save interaction matrix
        if 'interaction_matrix' in recommendation_data:
            matrix_path = os.path.join(version_dir, "interaction_matrix.npy")
            np.save(matrix_path, recommendation_data['interaction_matrix'])
            self.logger.info(f"Saved interaction matrix to {matrix_path}")
        
        # Save cooccurrence matrix
        if 'cooccurrence_matrix' in recommendation_data:
            matrix_path = os.path.join(version_dir, "cooccurrence_matrix.npy")
            np.save(matrix_path, recommendation_data['cooccurrence_matrix'])
            self.logger.info(f"Saved cooccurrence matrix to {matrix_path}")
            
        # Save customer profiles
        if 'customer_profiles' in recommendation_data:
            profiles_path = os.path.join(version_dir, "customer_profiles.json")
            # Convert datetime objects to strings for JSON serialization
            customer_profiles = self._prepare_dict_for_json(recommendation_data['customer_profiles'])
            with open(profiles_path, 'w') as f:
                json.dump(customer_profiles, f, indent=2)
            self.logger.info(f"Saved customer profiles to {profiles_path}")
            
        # Save product metadata
        if 'product_metadata' in recommendation_data:
            metadata_path = os.path.join(version_dir, "product_metadata.json")
            # Convert datetime objects to strings for JSON serialization
            product_metadata = self._prepare_dict_for_json(recommendation_data['product_metadata'])
            with open(metadata_path, 'w') as f:
                json.dump(product_metadata, f, indent=2)
            self.logger.info(f"Saved product metadata to {metadata_path}")
            
        # Save seasonal relevance scores
        if 'seasonal_relevance_scores' in recommendation_data:
            scores_path = os.path.join(version_dir, "seasonal_relevance_scores.json")
            with open(scores_path, 'w') as f:
                json.dump(recommendation_data['seasonal_relevance_scores'], f, indent=2)
            self.logger.info(f"Saved seasonal relevance scores to {scores_path}")
            
        # Save encodings
        encodings = {
            'customer_encodings': recommendation_data.get('customer_encodings', {}),
            'product_encodings': recommendation_data.get('product_encodings', {}),
            'reverse_customer_encodings': recommendation_data.get('reverse_customer_encodings', {}),
            'reverse_product_encodings': recommendation_data.get('reverse_product_encodings', {})
        }
        
        encodings_path = os.path.join(version_dir, "encodings.pickle")
        with open(encodings_path, 'wb') as f:
            pickle.dump(encodings, f)
        self.logger.info(f"Saved encodings to {encodings_path}")
        
        # Save version info
        version_info = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'data_contents': list(recommendation_data.keys())
        }
        
        version_path = os.path.join(version_dir, "version_info.json")
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Update latest version pointer
        latest_path = os.path.join(self.storage_dir, "latest_version.txt")
        with open(latest_path, 'w') as f:
            f.write(version)
            
        self.logger.info(f"Successfully saved all data for version {version}")
        return version_dir
    
    def load_processed_data(self, version=None):
        """
        Load processed data from storage
        
        Parameters:
        -----------
        version : str, optional
            Version identifier to load (default: latest version)
            
        Returns:
        --------
        dict
            Dictionary containing all loaded data
        """
        # Determine which version to load
        if version is None:
            # Try to load latest version
            latest_path = os.path.join(self.storage_dir, "latest_version.txt")
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    version = f.read().strip()
            else:
                self.logger.error("No latest version found")
                raise FileNotFoundError("No data has been saved yet")
        
        version_dir = os.path.join(self.storage_dir, f"version_{version}")
        if not os.path.exists(version_dir):
            self.logger.error(f"Version {version} not found")
            raise FileNotFoundError(f"Version {version} does not exist")
            
        self.logger.info(f"Loading data from version {version}")
        
        # Initialize results dictionary
        recommendation_data = {}
        
        # Load interaction matrix
        matrix_path = os.path.join(version_dir, "interaction_matrix.npy")
        if os.path.exists(matrix_path):
            recommendation_data['interaction_matrix'] = np.load(matrix_path)
            self.logger.info(f"Loaded interaction matrix from {matrix_path}")
        
        # Load cooccurrence matrix
        matrix_path = os.path.join(version_dir, "cooccurrence_matrix.npy")
        if os.path.exists(matrix_path):
            recommendation_data['cooccurrence_matrix'] = np.load(matrix_path)
            self.logger.info(f"Loaded cooccurrence matrix from {matrix_path}")
            
        # Load customer profiles
        profiles_path = os.path.join(version_dir, "customer_profiles.json")
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r') as f:
                recommendation_data['customer_profiles'] = json.load(f)
            self.logger.info(f"Loaded customer profiles from {profiles_path}")
            
        # Load product metadata
        metadata_path = os.path.join(version_dir, "product_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                recommendation_data['product_metadata'] = json.load(f)
            self.logger.info(f"Loaded product metadata from {metadata_path}")
            
        # Load seasonal relevance scores
        scores_path = os.path.join(version_dir, "seasonal_relevance_scores.json")
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                recommendation_data['seasonal_relevance_scores'] = json.load(f)
            self.logger.info(f"Loaded seasonal relevance scores from {scores_path}")
            
        # Load encodings
        encodings_path = os.path.join(version_dir, "encodings.pickle")
        if os.path.exists(encodings_path):
            with open(encodings_path, 'rb') as f:
                encodings = pickle.load(f)
                
            # Add encodings to results
            for key, value in encodings.items():
                recommendation_data[key] = value
                
            self.logger.info(f"Loaded encodings from {encodings_path}")
            
        self.logger.info(f"Successfully loaded data for version {version}")
        return recommendation_data
    
    def list_available_versions(self):
        """
        List all available data versions
        
        Returns:
        --------
        list
            List of version identifiers
        """
        versions = []
        
        # Look for version directories
        for item in os.listdir(self.storage_dir):
            if item.startswith("version_") and os.path.isdir(os.path.join(self.storage_dir, item)):
                version_id = item.replace("version_", "")
                versions.append(version_id)
                
        return sorted(versions)
    
    def get_version_info(self, version=None):
        """
        Get information about a specific version
        
        Parameters:
        -----------
        version : str, optional
            Version identifier (default: latest version)
            
        Returns:
        --------
        dict
            Dictionary containing version information
        """
        # Determine which version to check
        if version is None:
            # Try to get latest version
            latest_path = os.path.join(self.storage_dir, "latest_version.txt")
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    version = f.read().strip()
            else:
                self.logger.error("No latest version found")
                raise FileNotFoundError("No data has been saved yet")
        
        version_dir = os.path.join(self.storage_dir, f"version_{version}")
        if not os.path.exists(version_dir):
            self.logger.error(f"Version {version} not found")
            raise FileNotFoundError(f"Version {version} does not exist")
            
        # Load version info
        version_path = os.path.join(version_dir, "version_info.json")
        if os.path.exists(version_path):
            with open(version_path, 'r') as f:
                version_info = json.load(f)
            return version_info
        else:
            # If version info file doesn't exist, create basic info
            return {
                'version': version,
                'data_contents': [item.split('.')[0] for item in os.listdir(version_dir)]
            }
    
    def export_customer_data(self, output_path, version=None):
        """
        Export customer profiles to a CSV file
        
        Parameters:
        -----------
        output_path : str
            Path where the CSV file will be saved
        version : str, optional
            Version identifier (default: latest version)
            
        Returns:
        --------
        str
            Path to the exported CSV file
        """
        # Load customer profiles
        recommendation_data = self.load_processed_data(version)
        
        if 'customer_profiles' not in recommendation_data:
            self.logger.error("No customer profiles found in the loaded data")
            raise ValueError("Customer profiles are not available")
            
        customer_profiles = recommendation_data['customer_profiles']
        
        # Create a list of dictionaries for the DataFrame
        customer_data = []
        
        for customer_id, profile in customer_profiles.items():
            # Extract main customer metrics
            customer_row = {
                'customer_id': customer_id,
                'transaction_count': profile.get('transaction_count', 0),
                'unique_products': profile.get('unique_products', 0),
            }
            
            # Add temporal metrics if available
            if 'first_purchase' in profile:
                customer_row['first_purchase'] = profile['first_purchase']
                
            if 'last_purchase' in profile:
                customer_row['last_purchase'] = profile['last_purchase']
                
            if 'days_since_last_purchase' in profile:
                customer_row['days_since_last_purchase'] = profile['days_since_last_purchase']
                
            if 'avg_purchase_interval_days' in profile:
                customer_row['avg_purchase_interval_days'] = profile['avg_purchase_interval_days']
            
            # Add category preferences if available
            if 'top_categories' in profile and profile['top_categories']:
                for i, category in enumerate(profile['top_categories'][:3]):
                    customer_row[f'preferred_category_{i+1}'] = category
            
            customer_data.append(customer_row)
        
        # Create DataFrame
        df = pd.DataFrame(customer_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported customer data to {output_path}")
        
        return output_path
    
    def export_product_data(self, output_path, version=None):
        """
        Export product metadata to a CSV file
        
        Parameters:
        -----------
        output_path : str
            Path where the CSV file will be saved
        version : str, optional
            Version identifier (default: latest version)
            
        Returns:
        --------
        str
            Path to the exported CSV file
        """
        # Load product metadata
        recommendation_data = self.load_processed_data(version)
        
        if 'product_metadata' not in recommendation_data:
            self.logger.error("No product metadata found in the loaded data")
            raise ValueError("Product metadata is not available")
            
        product_metadata = recommendation_data['product_metadata']
        
        # Create a list of dictionaries for the DataFrame
        product_data = []
        
        for product_id, metadata in product_metadata.items():
            # Extract main product metrics
            product_row = {
                'product_id': product_id,
                'purchase_count': metadata.get('purchase_count', 0),
                'unique_customers': metadata.get('unique_customers', 0),
            }
            
            # Add category information if available
            if 'primary_category' in metadata:
                product_row['primary_category'] = metadata['primary_category']
            
            # Add price information if available
            if 'avg_price' in metadata:
                product_row['avg_price'] = metadata['avg_price']
                
            if 'min_price' in metadata:
                product_row['min_price'] = metadata['min_price']
                
            if 'max_price' in metadata:
                product_row['max_price'] = metadata['max_price']
            
            # Add temporal information if available
            if 'first_purchased' in metadata:
                product_row['first_purchased'] = metadata['first_purchased']
                
            if 'last_purchased' in metadata:
                product_row['last_purchased'] = metadata['last_purchased']
            
            # Add seasonality information
            if 'is_seasonal' in metadata:
                product_row['is_seasonal'] = metadata['is_seasonal']
                
            if 'peak_month' in metadata:
                product_row['peak_month'] = metadata['peak_month']
            
            # Add seasonal score if available
            if 'seasonal_relevance_scores' in recommendation_data:
                product_row['seasonal_relevance_score'] = recommendation_data['seasonal_relevance_scores'].get(product_id, 0.5)
            
            product_data.append(product_row)
        
        # Create DataFrame
        df = pd.DataFrame(product_data)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"Exported product data to {output_path}")
        
        return output_path
    
    def delete_version(self, version):
        """
        Delete a specific data version
        
        Parameters:
        -----------
        version : str
            Version identifier to delete
            
        Returns:
        --------
        bool
            True if deletion was successful
        """
        version_dir = os.path.join(self.storage_dir, f"version_{version}")
        
        if not os.path.exists(version_dir):
            self.logger.error(f"Version {version} not found")
            return False
            
        # Check if this is the latest version
        is_latest = False
        latest_path = os.path.join(self.storage_dir, "latest_version.txt")
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                latest_version = f.read().strip()
                if latest_version == version:
                    is_latest = True
        
        # Delete the version directory
        for root, dirs, files in os.walk(version_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir_name in dirs:
                os.rmdir(os.path.join(root, dir_name))
        os.rmdir(version_dir)
        
        self.logger.info(f"Deleted version {version}")
        
        # If we deleted the latest version, update the pointer
        if is_latest:
            versions = self.list_available_versions()
            if versions:
                # Set the most recent remaining version as latest
                new_latest = versions[-1]
                with open(latest_path, 'w') as f:
                    f.write(new_latest)
                self.logger.info(f"Updated latest version to {new_latest}")
            else:
                # No versions left, remove the latest pointer
                os.remove(latest_path)
                self.logger.info("Removed latest version pointer (no versions remain)")
        
        return True
    
    def _prepare_dict_for_json(self, data_dict):
        """
        Prepare a dictionary for JSON serialization by converting non-serializable objects
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary to prepare
            
        Returns:
        --------
        dict
            Dictionary with serializable values
        """
        if isinstance(data_dict, dict):
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    result[key] = self._prepare_dict_for_json(value)
                elif isinstance(value, (datetime, pd.Timestamp)):
                    result[key] = value.isoformat()
                elif isinstance(value, (list, tuple)):
                    result[key] = [self._prepare_dict_for_json(item) if isinstance(item, dict) else
                                  item.isoformat() if isinstance(item, (datetime, pd.Timestamp)) else item
                                  for item in value]
                elif isinstance(value, (int, float, str, bool, type(None))):
                    result[key] = value
                elif hasattr(value, 'tolist'):  # For numpy arrays
                    result[key] = value.tolist()
                else:
                    result[key] = str(value)
            return result
        return data_dict

# Example usage
if __name__ == "__main__":
    from DataMain import DataModule
    
    # Initialize modules
    data_module = DataModule()
    storage_module = StorageModule(storage_dir="recommendation_data")
    
    # Process data
    data_module.load_csv("transaction_data.csv")
    data_module.clean_data()
    recommendation_data = data_module.prepare_data_for_recommendations()
    
    # Save processed data
    storage_module.save_processed_data(recommendation_data)
    
    # Load processed data
    loaded_data = storage_module.load_processed_data()
    
    # Export data for analysis
    storage_module.export_customer_data("customer_analysis.csv")
    storage_module.export_product_data("product_analysis.csv")