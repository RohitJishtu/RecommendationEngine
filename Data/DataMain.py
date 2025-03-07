import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import logging
from collections import defaultdict

class DataModule:
    """
    Data Module for processing and preparing transaction data for recommendation models.
    Handles data loading, cleaning, feature engineering, and provides structured access
    to processed data.
    """
    
    def __init__(self):
        # Initialize data containers
        self.raw_data = None
        self.cleaned_data = None
        self.customer_encodings = {}
        self.product_encodings = {}
        self.customer_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataModule')
    
    def load_csv(self, file_path, encoding='utf-8'):
        """
        Load transaction data from a CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing transaction data
        encoding : str, optional
            Character encoding to use when reading the file (default: 'utf-8')
            
        Returns:
        --------
        pandas.DataFrame
            The loaded raw data
        """
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The file {file_path} does not exist")
        
        try:
            self.logger.info(f"Loading data from {file_path}")
            self.raw_data = pd.read_csv(file_path, encoding=encoding)
            self.logger.info(f"Loaded {len(self.raw_data)} transactions")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self):
        """
        Validate that the data contains the required columns
        
        Returns:
        --------
        bool
            True if data is valid, False otherwise
        """
        if self.raw_data is None:
            self.logger.error("No data loaded. Call load_csv() first.")
            return False
        
        # Define required columns
        required_columns = ['customer_id', 'product_id']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {', '.join(missing_columns)}")
            return False
            
        # Recommend optional columns
        recommended_columns = ['timestamp', 'product_category', 'purchase_amount']
        missing_recommended = [col for col in recommended_columns if col not in self.raw_data.columns]
        
        if missing_recommended:
            self.logger.warning(f"Missing recommended columns: {', '.join(missing_recommended)}")
        
        return True
    
    def clean_data(self):
        """
        Clean and preprocess the raw data
        
        Returns:
        --------
        pandas.DataFrame
            The cleaned data
        """
        if not self.validate_data():
            raise ValueError("Data validation failed. Cannot proceed with cleaning.")
        
        self.logger.info("Cleaning and preprocessing data")
        
        # Create a copy to avoid modifying the original DataFrame
        self.cleaned_data = self.raw_data.copy()
        
        # Remove duplicates
        initial_length = len(self.cleaned_data)
        self.cleaned_data.drop_duplicates(inplace=True)
        if initial_length > len(self.cleaned_data):
            self.logger.info(f"Removed {initial_length - len(self.cleaned_data)} duplicate rows")
        
        # Handle missing values
        for col in self.cleaned_data.columns:
            missing_count = self.cleaned_data[col].isna().sum()
            if missing_count > 0:
                self.logger.warning(f"Column '{col}' has {missing_count} missing values")
                
                # For critical columns, drop rows with missing values
                if col in ['customer_id', 'product_id']:
                    self.cleaned_data.dropna(subset=[col], inplace=True)
                    self.logger.info(f"Dropped {missing_count} rows with missing {col}")
        
        # Convert timestamp to datetime if available
        if 'timestamp' in self.cleaned_data.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(self.cleaned_data['timestamp']):
                    self.cleaned_data['timestamp'] = pd.to_datetime(self.cleaned_data['timestamp'])
                    
                # Add derived time features
                self.cleaned_data['day_of_week'] = self.cleaned_data['timestamp'].dt.dayofweek
                self.cleaned_data['month'] = self.cleaned_data['timestamp'].dt.month
                self.cleaned_data['year'] = self.cleaned_data['timestamp'].dt.year
                self.cleaned_data['hour'] = self.cleaned_data['timestamp'].dt.hour
                
                # Group by day (to identify transactions in same basket)
                self.cleaned_data['purchase_day'] = self.cleaned_data['timestamp'].dt.floor('D')
                
            except Exception as e:
                self.logger.error(f"Error converting timestamp: {str(e)}")
                # If conversion fails, drop the column to avoid issues
                self.cleaned_data.drop(columns=['timestamp'], inplace=True, errors='ignore')
        
        # Ensure all IDs are strings
        if 'customer_id' in self.cleaned_data.columns:
            self.cleaned_data['customer_id'] = self.cleaned_data['customer_id'].astype(str)
            
        if 'product_id' in self.cleaned_data.columns:
            self.cleaned_data['product_id'] = self.cleaned_data['product_id'].astype(str)
        
        self.logger.info(f"Data cleaning completed. {len(self.cleaned_data)} rows remain.")
        return self.cleaned_data
    
    def encode_categorical_features(self):
        """
        Encode categorical features like customer_id and product_id
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded features added
        """
        if self.cleaned_data is None:
            if self.raw_data is not None:
                self.clean_data()
            else:
                raise ValueError("No data loaded. Call load_csv() and clean_data() first.")
        
        self.logger.info("Encoding categorical features")
        
        # Encode customer_id
        self.cleaned_data['customer_id_encoded'] = self.customer_encoder.fit_transform(self.cleaned_data['customer_id'])
        
        # Encode product_id
        self.cleaned_data['product_id_encoded'] = self.product_encoder.fit_transform(self.cleaned_data['product_id'])
        
        # Create mappings for lookup
        self.customer_encodings = dict(zip(
            self.customer_encoder.classes_, 
            self.customer_encoder.transform(self.customer_encoder.classes_)
        ))
        
        self.product_encodings = dict(zip(
            self.product_encoder.classes_,
            self.product_encoder.transform(self.product_encoder.classes_)
        ))
        
        # Create reverse mappings
        self.reverse_customer_encodings = {v: k for k, v in self.customer_encodings.items()}
        self.reverse_product_encodings = {v: k for k, v in self.product_encodings.items()}
        
        self.logger.info(f"Encoded {len(self.customer_encodings)} unique customers and {len(self.product_encodings)} unique products")
        
        return self.cleaned_data
    
    def extract_product_metadata(self):
        """
        Extract metadata for products from transaction data
        
        Returns:
        --------
        dict
            Dictionary of product metadata
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Call clean_data() first.")
        
        self.logger.info("Extracting product metadata")
        
        product_metadata = {}
        
        # Get unique products
        unique_products = self.cleaned_data['product_id'].unique()
        
        for product_id in unique_products:
            product_data = self.cleaned_data[self.cleaned_data['product_id'] == product_id]
            
            metadata = {
                'product_id': product_id,
                'purchase_count': len(product_data),
                'unique_customers': product_data['customer_id'].nunique(),
            }
            
            # Include category if available
            if 'product_category' in self.cleaned_data.columns:
                categories = product_data['product_category'].unique()
                metadata['categories'] = categories.tolist()
                metadata['primary_category'] = product_data['product_category'].value_counts().index[0]
            
            # Include price if available
            if 'purchase_amount' in self.cleaned_data.columns:
                metadata['avg_price'] = product_data['purchase_amount'].mean()
                metadata['min_price'] = product_data['purchase_amount'].min()
                metadata['max_price'] = product_data['purchase_amount'].max()
            
            # Include temporal information if available
            if 'timestamp' in self.cleaned_data.columns:
                metadata['first_purchased'] = product_data['timestamp'].min()
                metadata['last_purchased'] = product_data['timestamp'].max()
                
                # Calculate purchase seasonality
                if 'month' in self.cleaned_data.columns:
                    month_counts = product_data['month'].value_counts().to_dict()
                    metadata['monthly_distribution'] = month_counts
                    
                    # Determine if product is seasonal
                    total_purchases = sum(month_counts.values())
                    monthly_percentages = {m: (count/total_purchases)*100 for m, count in month_counts.items()}
                    
                    # If any month has more than 30% of purchases, consider product seasonal
                    if any(pct > 30 for pct in monthly_percentages.values()):
                        metadata['is_seasonal'] = True
                        metadata['peak_month'] = max(monthly_percentages, key=monthly_percentages.get)
                    else:
                        metadata['is_seasonal'] = False
            
            product_metadata[product_id] = metadata
        
        self.product_metadata = product_metadata
        self.logger.info(f"Extracted metadata for {len(product_metadata)} products")
        
        return product_metadata
    
    def extract_customer_profiles(self):
        """
        Extract customer profiles from transaction data
        
        Returns:
        --------
        dict
            Dictionary of customer profiles
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Call clean_data() first.")
        
        self.logger.info("Extracting customer profiles")
        
        customer_profiles = {}
        
        # Get unique customers
        unique_customers = self.cleaned_data['customer_id'].unique()
        
        for customer_id in unique_customers:
            customer_data = self.cleaned_data[self.cleaned_data['customer_id'] == customer_id]
            
            profile = {
                'customer_id': customer_id,
                'transaction_count': len(customer_data),
                'unique_products': customer_data['product_id'].nunique(),
                'purchase_history': {}
            }
            
            # Include purchase history
            purchase_history = defaultdict(list)
            
            for _, row in customer_data.iterrows():
                product_id = row['product_id']
                
                product_entry = {'product_id': product_id}
                
                if 'timestamp' in row and pd.notna(row['timestamp']):
                    product_entry['timestamp'] = row['timestamp']
                
                if 'purchase_amount' in row and pd.notna(row['purchase_amount']):
                    product_entry['amount'] = row['purchase_amount']
                
                if 'product_category' in row and pd.notna(row['product_category']):
                    product_entry['category'] = row['product_category']
                
                purchase_history[product_id].append(product_entry)
            
            profile['purchase_history'] = dict(purchase_history)
            
            # Calculate category preferences if available
            if 'product_category' in self.cleaned_data.columns:
                category_counts = customer_data['product_category'].value_counts()
                top_categories = category_counts.head(3).index.tolist()
                profile['top_categories'] = top_categories
                profile['category_distribution'] = category_counts.to_dict()
            
            # Calculate purchase temporal patterns if available
            if 'timestamp' in self.cleaned_data.columns:
                profile['first_purchase'] = customer_data['timestamp'].min()
                profile['last_purchase'] = customer_data['timestamp'].max()
                profile['days_since_last_purchase'] = (datetime.now() - profile['last_purchase']).days
                
                # Calculate purchase frequency
                if len(customer_data) > 1:
                    purchase_dates = sorted(customer_data['timestamp'].unique())
                    intervals = [(purchase_dates[i] - purchase_dates[i-1]).days 
                                for i in range(1, len(purchase_dates))]
                    
                    if intervals:
                        profile['avg_purchase_interval_days'] = sum(intervals) / len(intervals)
                        profile['min_purchase_interval_days'] = min(intervals)
                        profile['max_purchase_interval_days'] = max(intervals)
                
                # Calculate purchase cycles for repeat products
                product_purchase_cycles = {}
                for product_id, purchases in purchase_history.items():
                    if 'timestamp' in purchases[0] and len(purchases) > 1:
                        dates = sorted([p['timestamp'] for p in purchases if 'timestamp' in p])
                        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
                        
                        if intervals:
                            product_purchase_cycles[product_id] = {
                                'avg_cycle_days': sum(intervals) / len(intervals),
                                'median_cycle_days': np.median(intervals),
                                'purchase_count': len(dates)
                            }
                
                if product_purchase_cycles:
                    profile['product_purchase_cycles'] = product_purchase_cycles
            
            customer_profiles[customer_id] = profile
        
        self.customer_profiles = customer_profiles
        self.logger.info(f"Extracted profiles for {len(customer_profiles)} customers")
        
        return customer_profiles
    
    def create_interaction_matrix(self):
        """
        Create a customer-product interaction matrix
        
        Returns:
        --------
        numpy.ndarray
            Matrix of customer-product interactions
        """
        if self.cleaned_data is None or 'customer_id_encoded' not in self.cleaned_data.columns:
            self.encode_categorical_features()
        
        self.logger.info("Creating customer-product interaction matrix")
        
        # Get matrix dimensions
        n_customers = len(self.customer_encodings)
        n_products = len(self.product_encodings)
        
        # Initialize interaction matrix
        interaction_matrix = np.zeros((n_customers, n_products))
        
        # Fill the matrix with interactions
        for _, row in self.cleaned_data.iterrows():
            customer_idx = row['customer_id_encoded']
            product_idx = row['product_id_encoded']
            
            # Apply different weighting schemes based on available data
            weight = 1.0  # Default weight is 1
            
            # If we have timestamps, use recency weighting
            if 'timestamp' in self.cleaned_data.columns:
                latest_purchase = self.cleaned_data['timestamp'].max()
                days_since = (latest_purchase - row['timestamp']).days
                # Apply decay factor based on recency
                recency_weight = np.exp(-0.05 * max(0, days_since))  # Adjust decay factor as needed
                weight *= recency_weight
            
            interaction_matrix[customer_idx, product_idx] += weight
        
        self.interaction_matrix = interaction_matrix
        self.logger.info(f"Created interaction matrix of shape {interaction_matrix.shape}")
        
        return interaction_matrix
    
    def create_product_cooccurrence_matrix(self):
        """
        Create a matrix of product co-occurrences in the same baskets
        
        Returns:
        --------
        numpy.ndarray
            Matrix of product co-occurrences
        """
        if self.cleaned_data is None or 'product_id_encoded' not in self.cleaned_data.columns:
            self.encode_categorical_features()
        
        self.logger.info("Creating product co-occurrence matrix")
        
        # Get matrix dimensions
        n_products = len(self.product_encodings)
        
        # Initialize co-occurrence matrix
        cooccurrence_matrix = np.zeros((n_products, n_products))
        
        # Group by basket
        if 'purchase_day' in self.cleaned_data.columns and 'customer_id' in self.cleaned_data.columns:
            # Use purchase_day to group items in the same basket
            baskets = self.cleaned_data.groupby(['customer_id', 'purchase_day'])
        else:
            # If no purchase_day, use customer_id alone
            baskets = self.cleaned_data.groupby('customer_id')
        
        # Count co-occurrences
        for _, basket in baskets:
            # Get unique products in this basket
            basket_products = basket['product_id_encoded'].unique()
            
            # Skip single-item baskets
            if len(basket_products) < 2:
                continue
            
            # Update co-occurrence counts
            for i in range(len(basket_products)):
                for j in range(len(basket_products)):
                    if i != j:  # Don't count product with itself
                        cooccurrence_matrix[basket_products[i], basket_products[j]] += 1
        
        # Normalize by row sums to get conditional probabilities
        row_sums = cooccurrence_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        cooccurrence_matrix = cooccurrence_matrix / row_sums
        
        self.cooccurrence_matrix = cooccurrence_matrix
        self.logger.info(f"Created co-occurrence matrix of shape {cooccurrence_matrix.shape}")
        
        return cooccurrence_matrix
    
    def get_seasonal_relevance_scores(self, reference_month=None):
        """
        Calculate seasonal relevance scores for products
        
        Parameters:
        -----------
        reference_month : int, optional
            Month to use as reference (1-12). If None, uses current month.
            
        Returns:
        --------
        dict
            Dictionary of product seasonal relevance scores
        """
        if not hasattr(self, 'product_metadata'):
            self.extract_product_metadata()
        
        self.logger.info("Calculating seasonal relevance scores")
        
        # Use current month if not specified
        if reference_month is None:
            reference_month = datetime.now().month
        
        seasonal_scores = {}
        
        for product_id, metadata in self.product_metadata.items():
            # Default score for products without seasonal data
            score = 0.5
            
            if 'monthly_distribution' in metadata:
                # Get monthly distribution
                monthly_dist = metadata['monthly_distribution']
                
                # Calculate similarity to reference month
                if reference_month in monthly_dist:
                    # Start with base score from actual month data
                    month_count = monthly_dist.get(reference_month, 0)
                    total_count = sum(monthly_dist.values())
                    
                    if total_count > 0:
                        # Calculate percentage of purchases in this month
                        month_percentage = month_count / total_count
                        
                        # Scale to 0-1 range (assuming max realistic percentage is 50%)
                        score = min(1.0, month_percentage * 2)
                        
                        # Boost score for very seasonal products
                        if metadata.get('is_seasonal', False) and metadata.get('peak_month') == reference_month:
                            score = min(1.0, score * 1.5)  # Boost seasonal items in their peak month
            
            seasonal_scores[product_id] = score
        
        self.seasonal_relevance_scores = seasonal_scores
        self.logger.info(f"Calculated seasonal relevance scores for {len(seasonal_scores)} products")
        
        return seasonal_scores
    
    def prepare_data_for_recommendations(self):
        """
        Prepare all necessary data structures for the recommendation system
        
        Returns:
        --------
        dict
            Dictionary containing all data needed for recommendations
        """
        self.logger.info("Preparing data for recommendation system")
        
        # Ensure all required data is processed
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
            
        if self.cleaned_data is None:
            self.clean_data()
            
        if 'customer_id_encoded' not in self.cleaned_data.columns:
            self.encode_categorical_features()
            
        if not hasattr(self, 'interaction_matrix'):
            self.create_interaction_matrix()
            
        if not hasattr(self, 'cooccurrence_matrix'):
            self.create_product_cooccurrence_matrix()
            
        if not hasattr(self, 'customer_profiles'):
            self.extract_customer_profiles()
            
        if not hasattr(self, 'product_metadata'):
            self.extract_product_metadata()
            
        if not hasattr(self, 'seasonal_relevance_scores'):
            self.get_seasonal_relevance_scores()
        
        # Compile all data into a single dictionary
        recommendation_data = {
            'interaction_matrix': self.interaction_matrix,
            'cooccurrence_matrix': self.cooccurrence_matrix,
            'customer_profiles': self.customer_profiles,
            'product_metadata': self.product_metadata,
            'seasonal_relevance_scores': self.seasonal_relevance_scores,
            'customer_encodings': self.customer_encodings,
            'product_encodings': self.product_encodings,
            'reverse_customer_encodings': self.reverse_customer_encodings,
            'reverse_product_encodings': self.reverse_product_encodings
        }
        
        self.logger.info("Data preparation complete")
        return recommendation_data
    
    def get_data_summary(self):
        """
        Get a summary of the data
        
        Returns:
        --------
        dict
            Dictionary containing data summary statistics
        """
        if self.cleaned_data is None:
            if self.raw_data is not None:
                self.clean_data()
            else:
                raise ValueError("No data loaded. Call load_csv() first.")
        
        summary = {
            'total_transactions': len(self.cleaned_data),
            'unique_customers': self.cleaned_data['customer_id'].nunique(),
            'unique_products': self.cleaned_data['product_id'].nunique(),
        }
        
        # Time span if available
        if 'timestamp' in self.cleaned_data.columns:
            summary['first_transaction'] = self.cleaned_data['timestamp'].min()
            summary['last_transaction'] = self.cleaned_data['timestamp'].max()
            summary['time_span_days'] = (summary['last_transaction'] - summary['first_transaction']).days
        
        # Category information if available
        if 'product_category' in self.cleaned_data.columns:
            summary['unique_categories'] = self.cleaned_data['product_category'].nunique()
            top_categories = self.cleaned_data['product_category'].value_counts().head(5)
            summary['top_categories'] = dict(zip(top_categories.index, top_categories.values))
        
        # Transaction value information if available
        if 'purchase_amount' in self.cleaned_data.columns:
            summary['total_purchase_value'] = self.cleaned_data['purchase_amount'].sum()
            summary['average_transaction_value'] = self.cleaned_data['purchase_amount'].mean()
            summary['max_transaction_value'] = self.cleaned_data['purchase_amount'].max()
        
        # Transactions per customer
        transactions_per_customer = self.cleaned_data.groupby('customer_id').size()
        summary['avg_transactions_per_customer'] = transactions_per_customer.mean()
        summary['max_transactions_per_customer'] = transactions_per_customer.max()
        summary['min_transactions_per_customer'] = transactions_per_customer.min()
        
        # Products per customer
        products_per_customer = self.cleaned_data.groupby('customer_id')['product_id'].nunique()
        summary['avg_unique_products_per_customer'] = products_per_customer.mean()
        summary['max_unique_products_per_customer'] = products_per_customer.max()
        
        # Customers per product
        customers_per_product = self.cleaned_data.groupby('product_id')['customer_id'].nunique()
        summary['avg_customers_per_product'] = customers_per_product.mean()
        summary['top_products'] = dict(zip(
            customers_per_product.sort_values(ascending=False).head(5).index,
            customers_per_product.sort_values(ascending=False).head(5).values
        ))
        
        return summary

# Example usage
if __name__ == "__main__":
    data_module = DataModule()
    
    # Load data
    data_module.load_csv("transaction_data.csv")
    
    # Clean and process data
    data_module.clean_data()
    data_module.encode_categorical_features()
    
    # Extract metadata and profiles
    product_metadata = data_module.extract_product_metadata()
    customer_profiles = data_module.extract_customer_profiles()
    
    # Create matrices for recommendations
    interaction_matrix = data_module.create_interaction_matrix()
    cooccurrence_matrix = data_module.create_product_cooccurrence_matrix()
    
    # Get seasonal relevance scores
    seasonal_scores = data_module.get_seasonal_relevance_scores()
    
    # Get data summary
    summary = data_module.get_data_summary()
    print("Data Summary:", summary)
    
    # Prepare all data for recommendations
    recommendation_data = data_module.prepare_data_for_recommendations()