import pandas as pd

orders = pd.read_csv('data/clean/orders.csv')
customers = pd.read_csv('data/clean/customers.csv')
products = pd.read_csv('data/clean/products.csv')

print(orders.head())
print(customers.head())
print(products.head())

orders = orders.merge(customers[['customer_id', 'city']], on='customer_id', how='left')
completed_orders = orders[orders['status']=='completed']
pivot = pd.pivot_table(
    completed_orders,
    index='city',
    columns='product_category',
    values='order_amount',
    aggfunc='sum',
    margins=True
)

print(f'Pivot table is\n:{pivot}')

print("Statistics:\n")
print(completed_orders['order_amount'].describe())
print(completed_orders['order_amount'].median())

columns = ['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']
correlation_matrix = completed_orders[columns].corr()
print(correlation_matrix)

print(f'Skewness:{completed_orders["order_amount"].skew()}')

stats = completed_orders['order_amount'].describe()
Q1 = stats['25%']
Q2 = stats['50%']
Q3 = stats['75%']
IQR = Q3 - Q1
Lower_boundary = Q1 - 1.5 * IQR
Upper_boundary = Q3 + 1.5 * IQR

#outliers
lower_outliers = (completed_orders['order_amount'] < Lower_boundary).sum()
upper_outliers = (completed_orders['order_amount'] > Upper_boundary).sum()

#results
print(f"IQR: {IQR:.2f}")
print(f"Lower Fence: {Lower_boundary:.2f}")
print(f"Upper Fence: {Upper_boundary:.2f}")
print(f"Lower Outliers: {lower_outliers}")
print(f"Upper Outliers: {upper_outliers}")