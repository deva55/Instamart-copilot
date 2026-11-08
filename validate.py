import pandas as pd

raw_orders = pd.read_csv('data/raw/orders.csv')
customers = pd.read_csv('data/clean/customers.csv')
orders = pd.read_csv("data/clean/orders.csv")
rfm = pd.read_csv("outputs/customer_segments.csv")


assert len(raw_orders) == 13084
assert raw_orders.duplicated().sum() == 119
assert (raw_orders['status'] == 'pending').sum() == 656
assert raw_orders['order_id'].nunique() == 12309
assert customers['city'].nunique() == 6
assert customers['age'].isna().sum() == 0
assert (customers['gender'] == 'female').sum() == 610
assert (customers['gender'] == 'male').sum() == 560

assert len(orders) == 12309
assert orders["order_id"].is_unique
assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199
assert (orders["status"] == "pending").sum() == 0
assert orders["order_amount"].isna().sum() == 0

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

assert completed["delivery_minutes"].isna().sum() == 0
assert cancelled["delivery_minutes"].isna().sum() == 1199
assert abs(completed["order_amount"].sum() - 2998668.80) < 0.01

print("ALL CHECKPOINTS PASSED")

assert len(rfm) == 1175
assert rfm["customer_id"].is_unique
assert rfm["segment_name"].notna().all()

# Calculate mean monetary value for each segment
segment_profile = rfm.groupby("segment_id").agg(
    mean_monetary=("monetary", "mean")
)

premium_cluster = segment_profile["mean_monetary"].idxmax()

# Verify Premium flag
assert (
    rfm["is_premium"] == (rfm["segment_id"] == premium_cluster)
).all()

# Verify Premium segment name
assert (
    rfm["segment_name"].eq("Premium") == rfm["is_premium"]
).all()

print("SEGMENT VALIDATION PASSED")