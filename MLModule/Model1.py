import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from datetime import datetime

class RecommendationModule:
    """
    Recommendation Module for generating personalized product recommendations
    based on processed customer transaction data.
    """
    
    def __init__(self, recommendation_data=None):
        """
        Initialize the recommendation module
        
        Parameters:
        -----------
        recommendation_data : dict, optional
            Dictionary containing processed data for recommendations
        """
        self.recommendation_data = recommendation_data
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RecommendationModule')
    
    def load_data(self, recommendation_data):
        """
        Load processed data for recommendations
        
        Parameters:
        -----------
        recommendation_data : dict
            Dictionary containing processed data
        """
        self.recommendation_data = recommendation_data
        self.logger.info("Loaded recommendation data")
        
        # Validate required data
        required_keys = [
            'interaction_matrix', 
            'cooccurrence_matrix',
            'customer_encodings',
            'product_encodings',
            'reverse_customer_encodings',
            'reverse_product_encodings'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.recommendation_data]
        
        if missing_keys:
            self.logger.warning(f"Missing recommended data keys: {', '.join(missing_keys)}")
    
    def get_similar_products(self, product_id, n=5, method='cooccurrence'):
        """
        Get products similar to a given product
        
        Parameters:
        -----------
        product_id : str
            ID of the product to find similar items for
        n : int, optional
            Number of similar products to return (default: 5)
        method : str, optional
            Method to use for finding similar products:
            - 'cooccurrence': Based on products bought together
            - 'content': Based on product metadata
            - 'hybrid': Combination of multiple methods
            
        Returns:
        --------
        list
            List of similar product IDs
        """
        if self.recommendation_data is None:
            raise ValueError("No recommendation data loaded")
            
        # Get product index
        if product_id not in self.recommendation_data['product_encodings']:
            self.logger.warning(f"Product {product_id} not found in the encodings")
            return []
            
        product_idx = self.recommendation_data['product_encodings'][product_id]
        similar_products = []
        
        if method == 'cooccurrence' or method == 'hybrid':
            # Use cooccurrence matrix
            if 'cooccurrence_matrix' in self.recommendation_data:
                # Get similarity scores from cooccurrence matrix
                similarity_scores = self.recommendation_data['cooccurrence_matrix'][product_idx]
                
                # Get top product indices
                top_indices = np.argsort(similarity_scores)[::-1][:n+1]  # +1 because the product itself will be included
                
                # Remove the product itself
                top_indices = top_indices[top_indices != product_idx][:n]
                
                # Convert indices to product IDs
                for idx in top_indices:
                    if idx in self.recommendation_data['reverse_product_encodings']:
                        similar_id = self.recommendation_data['reverse_product_encodings'][idx]
                        similar_products.append((similar_id, similarity_scores[idx]))
        
        if method == 'content' or method == 'hybrid':
            # Use product metadata for content-based similarity
            if 'product_metadata' in self.recommendation_data:
                product_metadata = self.recommendation_data['product_metadata']
                
                # Skip if the product is not in metadata
                if product_id not in product_metadata:
                    self.logger.warning(f"Product {product_id} not found in metadata")
                    if similar_products:  # If we already have recommendations from cooccurrence
                        similar_product_ids = [p[0] for p in similar_products[:n]]
                        return similar_product_ids
                    return []
                
                target_product = product_metadata[product_id]
                content_similarities = []
                
                # Calculate content similarity based on categories and other attributes
                for pid, metadata in product_metadata.items():
                    if pid == product_id:
                        continue  # Skip the target product
                        
                    similarity_score = 0.0
                    
                    # Category similarity
                    if 'primary_category' in target_product and 'primary_category' in metadata:
                        if target_product['primary_category'] == metadata['primary_category']:
                            similarity_score += 0.5
                    
                    # Price range similarity (if available)
                    if all(key in target_product for key in ['avg_price', 'min_price', 'max_price']) and \
                       all(key in metadata for key in ['avg_price', 'min_price', 'max_price']):
                        # Calculate price similarity based on overlap of price ranges
                        target_range = (target_product['min_price'], target_product['max_price'])
                        product_range = (metadata['min_price'], metadata['max_price'])
                        
                        # Check for overlap
                        if target_range[1] >= product_range[0] and product_range[1] >= target_range[0]:
                            # Calculate overlap percentage
                            overlap_min = max(target_range[0], product_range[0])
                            overlap_max = min(target_range[1], product_range[1])
                            overlap_size = overlap_max - overlap_min
                            
                            target_range_size = target_range[1] - target_range[0]
                            product_range_size = product_range[1] - product_range[0]
                            
                            if target_range_size > 0 and product_range_size > 0:
                                overlap_pct = overlap_size / max(target_range_size, product_range_size)
                                similarity_score += 0.3 * overlap_pct
                    
                    # Seasonality similarity (if available)
                    if 'is_seasonal' in target_product and 'is_seasonal' in metadata:
                        if target_product['is_seasonal'] == metadata['is_seasonal']:
                            similarity_score += 0.1
                            
                            # If both seasonal, check peak month
                            if target_product['is_seasonal'] and 'peak_month' in target_product and 'peak_month' in metadata:
                                if target_product['peak_month'] == metadata['peak_month']:
                                    similarity_score += 0.1
                    
                    content_similarities.append((pid, similarity_score))
                
                # Sort by similarity score
                content_similarities.sort(key=lambda x: x[1], reverse=True)
                
                if method == 'content':
                    similar_products = content_similarities[:n]
                else:  # hybrid
                    # Combine methods - add unique content-based recommendations
                    current_ids = set(p[0] for p in similar_products)
                    
                    for pid, score in content_similarities:
                        if pid not in current_ids and len(similar_products) < n:
                            similar_products.append((pid, score * 0.5))  # Weight content less than cooccurrence
                            current_ids.add(pid)
        
        # Sort final results by score
        similar_products.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the product IDs
        similar_product_ids = [p[0] for p in similar_products[:n]]
        return similar_product_ids
    
    def recommend_for_customer(self, customer_id, n=5, include_purchased=False):
        """
        Generate product recommendations for a specific customer
        
        Parameters:
        -----------
        customer_id : str
            ID of the customer to generate recommendations for
        n : int, optional
            Number of recommendations to generate (default: 5)
        include_purchased : bool, optional
            Whether to include products the customer has already purchased (default: False)
            
        Returns:
        --------
        list
            List of recommended product IDs
        """
        if self.recommendation_data is None:
            raise ValueError("No recommendation data loaded")
            
        # Check if customer exists
        if customer_id not in self.recommendation_data['customer_encodings']:
            self.logger.warning(f"Customer {customer_id} not found in the encodings")
            return []
            
        customer_idx = self.recommendation_data['customer_encodings'][customer_id]
        
        # Get customer's purchase history
        if 'interaction_matrix' in self.recommendation_data:
            customer_purchases = self.recommendation_data['interaction_matrix'][customer_idx]
            
            # Get products the customer has already purchased
            purchased_indices = np.where(customer_purchases > 0)[0]
            purchased_product_ids = [
                self.recommendation_data['reverse_product_encodings'][idx]
                for idx in purchased_indices
                if idx in self.recommendation_data['reverse_product_encodings']
            ]
            
            # Generate candidate recommendations using collaborative filtering
            candidate_scores = np.zeros(self.recommendation_data['interaction_matrix'].shape[1])
            
            # For each product the customer purchased
            for idx in purchased_indices:
                # Get similar products based on cooccurrence
                if 'cooccurrence_matrix' in self.recommendation_data:
                    similar_products = self.recommendation_data['cooccurrence_matrix'][idx]
                    
                    # Weight by how much the customer interacted with the product
                    weight = customer_purchases[idx]
                    candidate_scores += similar_products * weight
            
            # If we don't want to recommend products the customer already purchased
            if not include_purchased:
                candidate_scores[purchased_indices] = 0
                
            # Get top product indices
            top_indices = np.argsort(candidate_scores)[::-1][:n]
            
            # Convert indices to product IDs and scores
            recommendations = []
            for idx in top_indices:
                if idx in self.recommendation_data['reverse_product_encodings'] and candidate_scores[idx] > 0:
                    product_id = self.recommendation_data['reverse_product_encodings'][idx]
                    score = candidate_scores[idx]
                    recommendations.append((product_id, score))
            
            # Apply seasonal boosting if data is available
            if 'seasonal_relevance_scores' in self.recommendation_data:
                current_month = datetime.now().month
                seasonal_recommendations = []
                
                for product_id, base_score in recommendations:
                    # Apply seasonal relevance modifier
                    seasonal_score = self.recommendation_data['seasonal_relevance_scores'].get(product_id, 0.5)
                    final_score = base_score * (0.7 + 0.6 * seasonal_score)  # Scale from 0.7-1.3 based on seasonality
                    seasonal_recommendations.append((product_id, final_score))
                
                # Re-sort based on seasonally adjusted scores
                seasonal_recommendations.sort(key=lambda x: x[1], reverse=True)
                recommendations = seasonal_recommendations
            
            # Return just the product IDs
            recommended_product_ids = [p[0] for p in recommendations]
            return recommended_product_ids
            
        # Fallback to content-based recommendations if no interaction data
        elif 'customer_profiles' in self.recommendation_data and customer_id in self.recommendation_data['customer_profiles']:
            customer_profile = self.recommendation_data['customer_profiles'][customer_id]
            
            # Get customer's past purchases
            purchased_products = list(customer_profile.get('purchase_history', {}).keys())
            
            # Get their top categories if available
            top_categories = customer_profile.get('top_categories', [])
            
            # Generate candidate recommendations by finding similar products to past purchases
            candidate_products = []
            
            for product_id in purchased_products:
                similar_products = self.get_similar_products(product_id, n=3, method='content')
                candidate_products.extend(similar_products)
            
            # Filter out already purchased products if requested
            if not include_purchased:
                candidate_products = [p for p in candidate_products if p not in purchased_products]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = [p for p in candidate_products if not (p in seen or seen.add(p))]
            
            return unique_candidates[:n]
        
        else:
            self.logger.warning(f"No interaction data or customer profile available for {customer_id}")
            return []
    
    def get_trending_products(self, n=10, timeframe_days=30):
        """
        Get trending products based on recent popularity
        
        Parameters:
        -----------
        n : int, optional
            Number of trending products to return (default: 10)
        timeframe_days : int, optional
            Number of days to consider for trending analysis (default: 30)
            
        Returns:
        --------
        list
            List of trending product IDs
        """
        if self.recommendation_data is None:
            raise ValueError("No recommendation data loaded")
            
        # Check if we have the necessary data
        if 'product_metadata' not in self.recommendation_data:
            self.logger.warning("Product metadata not available for trending analysis")
            return []
        
        product_metadata = self.recommendation_data['product_metadata']
        
        # Calculate trending score for each product
        trending_scores = {}
        
        for product_id, metadata in product_metadata.items():
            # Initialize score
            score = 0
            
            # Use purchase count as base score
            if 'purchase_count' in metadata:
                score = metadata['purchase_count']
            
            # Boost score for products with recent purchases
            if 'last_purchased' in metadata:
                try:
                    # Parse last_purchased if it's a string
                    if isinstance(metadata['last_purchased'], str):
                        last_purchased = datetime.fromisoformat(metadata['last_purchased'].replace('Z', '+00:00'))
                    else:
                        last_purchased = metadata['last_purchased']
                    
                    # Calculate days since last purchase
                    days_since = (datetime.now() - last_purchased).days
                    
                    # Apply recency boost (more recent = higher score)
                    if days_since <= timeframe_days:
                        recency_factor = 1 + 2 * (1 - days_since / timeframe_days)  # 1-3x multiplier
                        score *= recency_factor
                except (ValueError, TypeError):
                    pass
            
            # Boost score for products with high customer count
            if 'unique_customers' in metadata:
                score *= (1 + 0.1 * min(10, metadata['unique_customers']))  # Up to 2x multiplier
            
            # Apply seasonal boost
            if 'seasonal_relevance_scores' in self.recommendation_data:
                seasonal_score = self.recommendation_data['seasonal_relevance_scores'].get(product_id, 0.5)
                score *= (0.8 + 0.4 * seasonal_score)  # 0.8-1.2x multiplier
            
            trending_scores[product_id] = score
        
        # Sort products by trending score
        sorted_products = sorted(trending_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N trending product IDs
        return [p[0] for p in sorted_products[:n]]
    
    def get_personalized_trending(self, customer_id, n=5):
        """
        Get personalized trending products for a specific customer
        
        Parameters:
        -----------
        customer_id : str
            ID of the customer to generate trending recommendations for
        n : int, optional
            Number of recommendations to generate (default: 5)
            
        Returns:
        --------
        list
            List of personalized trending product IDs
        """
        if self.recommendation_data is None:
            raise ValueError("No recommendation data loaded")
            
        # Get general trending products
        trending_products = self.get_trending_products(n=20)
        
        # If customer doesn't exist, return general trending
        if customer_id not in self.recommendation_data.get('customer_encodings', {}):
            return trending_products[:n]
        
        # Get customer profile
        customer_profile = self.recommendation_data.get('customer_profiles', {}).get(customer_id, {})
        
        # Get customer's preferred categories
        preferred_categories = customer_profile.get('top_categories', [])
        
        # Score trending products based on relevance to customer preferences
        scored_products = []
        
        for product_id in trending_products:
            # Get product metadata
            metadata = self.recommendation_data.get('product_metadata', {}).get(product_id, {})
            
            # Initialize relevance score
            relevance = 1.0
            
            # Boost score for products in preferred categories
            if 'primary_category' in metadata and preferred_categories:
                if metadata['primary_category'] in preferred_categories:
                    category_index = preferred_categories.index(metadata['primary_category'])
                    category_boost = 2.0 / (category_index + 1)  # Higher boost for higher ranked categories
                    relevance *= category_boost
            
            # Apply price range preference if available
            if 'avg_price' in metadata and 'purchase_history' in customer_profile:
                # Calculate customer's average purchase amount
                purchase_amounts = []
                for purchases in customer_profile['purchase_history'].values():
                    for purchase in purchases:
                        if 'amount' in purchase:
                            purchase_amounts.append(purchase['amount'])
                
                if purchase_amounts:
                    avg_purchase = sum(purchase_amounts) / len(purchase_amounts)
                    product_price = metadata['avg_price']
                    
                    # Calculate price similarity (higher similarity = higher score)
                    if avg_purchase > 0:
                        price_ratio = min(product_price / avg_purchase, avg_purchase / product_price)
                        price_similarity = min(1.0, price_ratio)  # 0-1 range
                        relevance *= (0.5 + 0.5 * price_similarity)  # 0.5-1.0x multiplier
            
            scored_products.append((product_id, relevance))
        
        # Sort by relevance score
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N personalized trending products
        return [p[0] for p in scored_products[:n]]