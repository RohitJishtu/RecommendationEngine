import random
import csv
from datetime import datetime, timedelta

# Generate unique customers and products
num_customers = 1000  # Adjust as needed
num_products = 100  # 100 unique products

customers = [f"C{str(i).zfill(4)}" for i in range(1, num_customers + 1)]
products = {f"P{str(i).zfill(3)}": random.choice(["Electronics", "Accessories", "Clothing", "Home", "Food"]) for i in range(1, num_products + 1)}
price_ranges = {
    "Electronics": (400, 600),
    "Accessories": (10, 50),
    "Clothing": (30, 100),
    "Home": (50, 200),
    "Food": (5, 20)
}

def generate_random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

def generate_transactions(num_records):
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 31)
    transactions = []
    
    for _ in range(num_records):
        customer_id = random.choice(customers)
        product_id, category = random.choice(list(products.items()))
        timestamp = generate_random_timestamp(start_date, end_date).strftime('%Y-%m-%d %H:%M:%S')
        purchase_amount = round(random.uniform(*price_ranges[category]), 2)
        transactions.append([customer_id, product_id, timestamp, category, purchase_amount])
    
    return transactions

# Generate data
num_records = 10000  # Generate 10,000 transactions
data = generate_transactions(num_records)

# Save to CSV
filename = "transaction_data.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["customer_id", "product_id", "timestamp", "product_category", "purchase_amount"])
    writer.writerows(data)

print(f"Generated {num_records} transactions and saved to {filename}.")
