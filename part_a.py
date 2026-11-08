import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

#WORKING_DIR = '***Capestone Project/Dataset/data/raw/' - commenting the path for data masking

#customer_path = pd.read_csv(WORKING_DIR + 'customers.csv')
#product_path = pd.read_csv(WORKING_DIR + 'products.csv')
#order_path = pd.read_csv(WORKING_DIR + 'orders.csv')

#customer_df = pd.DataFrame(customer_path)
#product_df = pd.DataFrame(product_path)
#order_df = pd.DataFrame(order_path)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

customer_df = pd.read_csv(RAW_DIR / "customers.csv")
product_df = pd.read_csv(RAW_DIR / "products.csv")
order_df = pd.read_csv(RAW_DIR / "orders.csv")

print("Customer DataFrame before Datacleaning:")
print(customer_df.head())
print(f"Total no. of Blank ages:{customer_df['age'].isna().sum()}")
print(f"Ages outside 10-100:{((~customer_df['age'].between(10, 100)) & (customer_df['age'].notna())).sum()}")
print(f"Raw city spellings:{customer_df['city'].nunique()}")
print(f"different gender spellings:{customer_df['gender'].nunique()}")
print(f"Null gender counts:{customer_df['gender'].isnull().sum()}")

#3
customer_df['city'] = customer_df['city'].str.lower().str.strip().str.capitalize()
print(customer_df['city'].value_counts())

customer_df['gender'] = customer_df['gender'].astype('string').str.strip().str.lower()
gender_map = {'m': 'male', 'f': 'female', 'female': 'female', 'male' : 'male'}
customer_df['gender'] = customer_df['gender'].map(gender_map)
customer_df['gender'] = customer_df['gender'].fillna('unknown')
print(customer_df['gender'].value_counts())

print(customer_df['age'].isnull().sum())
customer_df.loc[(customer_df['age'] <= 10) | (customer_df['age'] > 100), 'age'] = np.nan
print(customer_df['age'].isnull().sum())
customer_df['age'] = customer_df['age'].fillna(customer_df.groupby('city')['age'].transform('median'))
#customer_df['age'] = customer_df['age'].fillna(customer_df['age'].median())
customer_df['age'] = customer_df['age'].round().astype(int)
print(customer_df['age'].isnull().sum())
customer_df['signup_date'] = pd.to_datetime(customer_df['signup_date'], format='%Y-%m-%d', errors='coerce')
print(customer_df[['city', 'gender', 'age', 'signup_date']].isna().sum())

#4.1
print("Order DataFrame before Datacleaning:")
print(order_df.head())
print(f"Total raw order rows: {len(order_df)}")
print(f"Exact duplicated rows:{order_df.duplicated().sum()}")
order_df.drop_duplicates(inplace=True)
print(f"Exact duplicates after drop:{order_df.duplicated().sum()}")
print(f"Pending orders:{(order_df['status'] == 'pending').sum()}")
print(f"Distinct order-ids:{order_df['order_id'].nunique()}")
#order_df['order_amount'] = pd.to_numeric(order_df['order_amount'], errors='coerce')
#order_df['null_order_amount'] = order_df['order_amount'].isna()
#print(f'Non-numeric amounts: {order_df['null_order_amount'].sum()}')
non_numeric = pd.to_numeric(order_df['order_amount'], errors='coerce').isna()
print(f"Non-numeric amounts: {non_numeric.sum()}")
print(f"Negative delivery times:{(order_df['delivery_minutes'] < 0 ).sum()}")
print(f"Blank delivery times:{order_df['delivery_minutes'].isna().sum()}")

print(order_df.info())

#4.2
order_df['order_ts'] = pd.to_datetime(order_df['order_ts'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
order_df.sort_values('order_ts', inplace=True)
order_df = order_df.drop_duplicates(subset=['order_id'], keep='last')
print(order_df['order_ts'].dtype)
print(order_df['order_id'].duplicated().sum())
order_df['price_inr'] = order_df['product_id'].map(product_df.set_index('product_id')['price_inr'])
order_df['order_amount'] = pd.to_numeric(order_df['order_amount'], errors='coerce')
order_df['order_amount'] = order_df['order_amount'].fillna(
    (
        order_df['price_inr'] *
        order_df['quantity'] *
        (1 - order_df['discount_pct'] / 100)
    ).round(2)
)
order_df.loc[(order_df['delivery_minutes'] < 0),  'delivery_minutes'] = np.nan
completed = order_df['status'] == 'completed'

order_df['product_category'] = order_df['product_id'].map(product_df.set_index('product_id')['category'])
order_df.loc[completed, 'delivery_minutes'] = (
    order_df.loc[completed, 'delivery_minutes'].fillna(
        order_df.loc[completed]
        .groupby('product_category')['delivery_minutes']
        .transform('median')
    )
)

#5
#DATA_DIR = Path("****/Capestone Project/Dataset/data") -- commenting for data masking

#CLEAN_DIR = DATA_DIR / "clean"

# Create folder if it doesn't exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Save cleaned DataFrames
customer_df.to_csv(CLEAN_DIR / "customers.csv", index=False)
product_df.to_csv(CLEAN_DIR / "products.csv", index=False)
order_df.to_csv(CLEAN_DIR / "orders.csv", index=False)

#6
# Create a copy for database export
orders_sql = order_df.copy()

# Convert timestamp to required text format
orders_sql['order_ts'] = orders_sql['order_ts'].dt.strftime(
    '%Y-%m-%d %H:%M:%S'
)

# Create or connect to database
DB_PATH = DATA_DIR / "instamart.db"

with sqlite3.connect(DB_PATH) as conn:

    customer_df.to_sql(
        'customers', conn, if_exists='replace', index=False
    )

    product_df.to_sql(
        'products', conn, if_exists='replace', index=False
    )

    orders_sql.to_sql(
        'orders', conn, if_exists='replace', index=False
    )


print("Cleaned CSV and SQLite exports completed.")



'''
print("Customer DataFrame:")
print(customer_df.head())
print(customer_df.info())
print(customer_df.describe())

print("\nMissing values Before:")


print("\nProduct DataFrame:")
print(product_df.head(40))
print(product_df.info())
print(product_df.describe())

print("\nOrder DataFrame:")
print(order_df.head())
print(order_df.info())
print(order_df.describe())








print(customer_df['gender'].value_counts())





print(customer_df[['city', 'gender', 'age', 'signup_date']].head(20))
print("\nMissing values:")
print(customer_df[['city', 'gender', 'age', 'signup_date']].isna().sum())

print("\nData types:")
print(customer_df.dtypes)
'''

