'''
The retention team wants two things from the clean order history: named customer segments to design campaigns around, 
and a lapse model that scores how likely a customer is to stop ordering. This Part consumes data/clean/orders.csv and data/clean/customers.csv from Part A, 
and its segment file feeds Part C's ticket routing. Use only completed orders for customer behaviour unless a task says otherwise.

Tasks

1.Build the RFM table (3 marks). For every customer with at least one completed order, compute, as of the snapshot moment 2025-10-01 00:00:00: 
recency = whole days between the snapshot and the customer's most recent completed order ((snapshot − last_order_ts).dt.days); 
frequency = number of distinct completed orders; monetary = sum of completed order_amount. 
Reference: 1,175 customers; recency median 19, minimum 0, maximum 180; frequency mean 9.46, maximum 33; monetary mean 2,552.06, maximum 14,766.55.
'''

import pandas as pd
import numpy as np
from pathlib import Path

orders = pd.read_csv('data/clean/orders.csv')
customers = pd.read_csv('data/clean/customers.csv')
products = pd.read_csv('data/clean/products.csv')

#print(orders.dtypes)

completed_orders = orders[orders['status']=='completed'].copy()
snapshot = pd.Timestamp('2025-10-01 00:00:00')
completed_orders['order_ts'] = pd.to_datetime(completed_orders['order_ts'])
completed_orders.loc[:, 'recency'] = (snapshot - completed_orders.groupby('customer_id')['order_ts'].transform('max')).dt.days
completed_orders['frequency'] = (completed_orders.groupby('customer_id')['order_id'].transform('nunique'))
completed_orders['monetary'] = ( completed_orders.groupby('customer_id')['order_amount'].transform('sum'))
rfm = completed_orders[
    ['customer_id', 'recency', 'frequency', 'monetary']
].drop_duplicates(subset='customer_id')
print(rfm.shape)
print(rfm[['recency', 'frequency', 'monetary']].describe())
print(rfm.head())

'''
Cluster with K-means (5 marks). Standardise the three RFM columns with StandardScaler. 
For every K from 2 to 8 fit KMeans(n_clusters=K, n_init=10, random_state=42) and record the inertia and the silhouette score. 
Present the table, read the elbow, and choose the final K from {3, 4, 5} using the elbow, the silhouette scores and business interpretability 
(a number that the business can act on beats a marginally better metric). 
Reference (tolerance ±5% on inertia, ±0.05 on silhouette): K=2: 1820.8 / 0.465; K=3: 1268.2 / 0.445; K=4: 743.1 / 0.526; K=5: 617.4 / 0.481; K=6: 503.2 / 0.482; K=7: 408.5 / 0.416; K=8: 366.5 / 0.412. 
Explain in two or three sentences why the inertia of K=8 is always lower than that of K=2 yet that fact alone cannot pick K
'''
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = rfm[['recency', 'frequency', 'monetary']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])
print(X_scaled.shape)

results = []

for k in range(2, 9):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)

    labels = model.fit_predict(X_scaled)

    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append([k, inertia, silhouette])

results_df = pd.DataFrame(
    results,
    columns=['K', 'Inertia', 'Silhouette']
)

print(results_df)

'''
Profile, name and project the segments (3 marks). (a) For the chosen K, report each segment's size and its mean recency, frequency and monetary values. 
(b) Give each segment a business name and one recommended action, and mark as Premium the segment with the highest mean monetary value (whatever K you chose). 
Remind the reader that cluster numbers are only labels with no ordering. Reference for K=4: sizes about 436 / 138 / 419 / 182; 
the 138-customer segment has mean monetary about 7,130 and mean frequency about 6.8; the 419-customer segment has mean frequency about 19.4 and mean recency about 8.8 days. 
(c) Apply PCA(n_components=2) to the scaled RFM matrix and report explained_variance_ratio_ (reference: about 0.634 and 0.218, total about 0.852, tolerance ±0.02). 
Write outputs/customer_segments.csv with columns customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium.

--To visualize the K-means elbow method, you can use the following code snippet:
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

plt.plot(results_df['K'], results_df['Inertia'], marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(range(2, 9))
plt.grid(True)
plt.show()
'''
final_k = 4
final_model = KMeans(
    n_clusters=final_k,
    n_init=10,
    random_state=42
)

rfm['cluster'] = final_model.fit_predict(X_scaled)

print(rfm[['customer_id', 'recency', 'frequency', 'monetary', 'cluster']].head())
print(rfm['cluster'].value_counts().sort_index())

segment_profile = rfm.groupby('cluster').agg(
    size=('customer_id', 'count'),
    mean_recency=('recency', 'mean'),
    mean_frequency=('frequency', 'mean'),
    mean_monetary=('monetary', 'mean')
)
print(segment_profile)

segment_names = {
    0: 'Premium',
    1: 'Lapsing',
    2: 'Active',
    3: 'Occasional'
}

rfm['segment_name'] = rfm['cluster'].map(segment_names)
premium_cluster = segment_profile['mean_monetary'].idxmax()
rfm['is_premium'] = rfm['cluster'] == premium_cluster

print(f'RFM:\n{rfm.head(30)}')


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print("PCA shape:", X_pca.shape)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

rfm = rfm.rename(columns={"cluster": "segment_id"})

rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
].to_csv(output_dir / "customer_segments.csv", index=False)

print("Customer segments exported successfully.")

'''
4. Define the lapse problem (3 marks). Use the cutoff 2025-09-01. The cohort is every customer with at least one completed order before the cutoff. 
The label lapsed is 1 if the customer has no completed order dated 2025-09-01 through 2025-09-30, otherwise 0. 
Build, using only orders dated before the cutoff plus the clean customer table, at least these features: recency (days from the cutoff to the last completed order), 
frequency, monetary, average discount_pct of completed orders, cancellation rate (cancelled orders ÷ all orders with a final status, before the cutoff), 
average delivery_minutes of completed orders, most frequent product category, city, age, gender, and tenure in days (cutoff minus signup_date). 
You must not use any order dated on or after the cutoff and must not use the segment labels from task 3 as features, because both would leak the answer. 
Split with train_test_split(test_size=0.2, stratify=y, random_state=42). Reference: 1,156 customers in the cohort; 443 lapsed (38.32%); 924 training and 232 test rows; 
always predicting "not lapsed" on the test set gives accuracy 61.6% and a lapsed-class F1 of 0 (pass zero_division=0 to the scikit-learn metric functions to silence the warning).
'''
cutoff = pd.Timestamp('2025-09-01')

orders['order_ts'] = pd.to_datetime(orders['order_ts'])

# Orders available before the cutoff
historical_orders = orders[
    orders['order_ts'] < cutoff
].copy()

# September completed orders — used ONLY for the target label
september_completed = orders[
    (orders['order_ts'] >= cutoff) &
    (orders['order_ts'] < pd.Timestamp('2025-10-01')) &
    (orders['status'] == 'completed')
].copy()

historical_completed = historical_orders[
    historical_orders['status'] == 'completed'
].copy()
cohort = historical_completed[['customer_id']].drop_duplicates()
print("Cohort size:", len(cohort))

september_customer_ids = set(september_completed['customer_id'])
cohort['lapsed'] = (
    ~cohort['customer_id'].isin(september_customer_ids)
).astype(int)
print(cohort['lapsed'].value_counts())

# 1. Calculate historical RFM and average values
customer_features = historical_completed.groupby('customer_id').agg(
    last_order=('order_ts', 'max'),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum'),
    avg_discount=('discount_pct', 'mean'),
    avg_delivery=('delivery_minutes', 'mean')
).reset_index()

customer_features['recency'] = (
    cutoff - customer_features['last_order']
).dt.days

customer_features = customer_features.drop(columns=['last_order']) #to verify the recency count


# 2. Calculate cancellation rate
final_orders = historical_orders[
    historical_orders['status'].isin(['completed', 'cancelled'])
].copy()

final_orders['is_cancelled'] = (
    final_orders['status'] == 'cancelled'
).astype(int)

cancellation = final_orders.groupby('customer_id').agg(
    cancellation_rate=('is_cancelled', 'mean')
).reset_index()

customer_features = customer_features.merge(
    cancellation,
    on='customer_id',
    how='left'
)


# 3. Find each customer's most frequent product category
orders_with_category = historical_completed.merge(
    products[['product_id', 'category']],
    on='product_id',
    how='left'
)

favorite_category = (
    orders_with_category.groupby('customer_id')['category']
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
    .reset_index(name='favorite_category')
)

customer_features = customer_features.merge(
    favorite_category,
    on='customer_id',
    how='left'
)


# 4. Calculate customer tenure
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

customers['tenure_days'] = (
    cutoff - customers['signup_date']
).dt.days


# 5. Add city, age, gender and tenure
customer_features = customer_features.merge(
    customers[['customer_id', 'city', 'age', 'gender', 'tenure_days']],
    on='customer_id',
    how='left'
)


# 6. Add the lapse target
customer_features = customer_features.merge(
    cohort[['customer_id', 'lapsed']],
    on='customer_id',
    how='left'
)


# 7. Verify results
print(customer_features.shape)
print(customer_features.head())
print(customer_features.isnull().sum())
print(customer_features['lapsed'].value_counts())


from sklearn.model_selection import train_test_split

X = customer_features.drop(columns=['customer_id', 'lapsed'])
y = customer_features['lapsed']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training:", X_train.shape)
print("Testing:", X_test.shape)
print("Training labels:", y_train.value_counts())
print("Testing labels:", y_test.value_counts())

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict 0 (not lapsed) for every test customer
baseline_pred = np.zeros(len(y_test), dtype=int)

print("Accuracy:", accuracy_score(y_test, baseline_pred))
print("Precision:", precision_score(y_test, baseline_pred, zero_division=0))
print("Recall:", recall_score(y_test, baseline_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, baseline_pred, zero_division=0))

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = [
    'recency',
    'frequency',
    'monetary',
    'avg_discount',
    'avg_delivery',
    'cancellation_rate',
    'age',
    'tenure_days'
]

categorical_features = [
    'favorite_category',
    'city',
    'gender'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

train_frequency_mean = (
    preprocessor.named_transformers_['num'].mean_[
        numeric_features.index('frequency')
    ]
)

whole_cohort_frequency_mean = X['frequency'].mean()

print("Training scaler mean frequency:", train_frequency_mean)
print("Whole cohort mean frequency:", whole_cohort_frequency_mean)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)
y_pred = lr_pipeline.predict(X_test)
print(y_pred[:10])

from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100]
}

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

print("Best parameters:", lr_grid.best_params_)
print("Best CV F1 score:", lr_grid.best_score_)

lr_results = pd.DataFrame(lr_grid.cv_results_)

print(
    lr_results[
        ['param_model__C', 'mean_test_score', 'std_test_score']
    ]
)

#retriving best fitted pipeline
best_lr = lr_grid.best_estimator_

#comparing the same with random forest
from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_param_grid = {
    'model__max_depth': [3, 5, 8],
    'model__min_samples_leaf': [5, 10, 20]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best RF parameters:", rf_grid.best_params_)
print("Best RF CV F1:", rf_grid.best_score_)

rf_results = pd.DataFrame(rf_grid.cv_results_)

print(
    rf_results[
        [
            'param_model__max_depth',
            'param_model__min_samples_leaf',
            'mean_test_score'
        ]
    ]
)

best_rf = rf_grid.best_estimator_

best_lr = lr_grid.best_estimator_
best_rf = rf_grid.best_estimator_

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

for name, model in [
    ('Logistic Regression', best_lr),
    ('Random Forest', best_rf)
]:
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

from sklearn.utils import resample
from sklearn.base import clone

minority_X = X_train[y_train == 1]
minority_y = y_train[y_train == 1]

majority_count = (y_train == 0).sum()
additional_needed = majority_count - len(minority_X)

# Draw only the additional 216 duplicates
extra_indices = resample(
    minority_X.index.to_numpy(),
    replace=True,
    n_samples=additional_needed,
    random_state=42
)

X_balanced = pd.concat([
    X_train,
    X_train.loc[extra_indices]
], ignore_index=True)

y_balanced = pd.concat([
    y_train,
    y_train.loc[extra_indices]
], ignore_index=True)

print(y_balanced.value_counts())

balanced_lr = clone(lr_grid.best_estimator_)
balanced_lr.fit(X_balanced, y_balanced)

from sklearn.metrics import precision_score, recall_score, f1_score

original_pred = lr_grid.best_estimator_.predict(X_test)
balanced_pred = balanced_lr.predict(X_test)

comparison = []

for name, pred in [
    ('Original Logistic Regression', original_pred),
    ('Balanced Logistic Regression', balanced_pred)
]:
    comparison.append({
        'Model': name,
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0)
    })

comparison_df = pd.DataFrame(comparison)

print(comparison_df)

change = (
    comparison_df.loc[1, ['Precision', 'Recall', 'F1']]
    - comparison_df.loc[0, ['Precision', 'Recall', 'F1']]
)

print("Metric changes:")
print(change)

print("Largest absolute change:", change.abs().idxmax())

# Extract the fitted Random Forest classifier from the pipeline
rf_model = best_rf.named_steps['model']

# Retrieve feature names after preprocessing
feature_names = best_rf.named_steps['preprocessor'].get_feature_names_out()

# Create a feature importance table
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})

# Sort from highest to lowest importance
top_5 = feature_importance.sort_values(
    by='importance',
    ascending=False
).head(5)

print(top_5)

# Extract trained Logistic Regression
lr_model = best_lr.named_steps['model']

# Retrieve names and coefficients
lr_feature_names = best_lr.named_steps['preprocessor'].get_feature_names_out()

coefficients = pd.DataFrame({
    'feature': lr_feature_names,
    'coefficient': lr_model.coef_[0]
})

# Select three numerical features for interpretation
selected_coefficients = coefficients[
    coefficients['feature'].isin([
        'num__recency',
        'num__frequency',
        'num__monetary'
    ])
].copy()

selected_coefficients['sign'] = selected_coefficients[
    'coefficient'
].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Zero')

print(selected_coefficients)

import joblib
import os
import numpy as np

# Create outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Select the final model
selected_model = best_lr

# Save the complete pipeline
joblib.dump(selected_model, 'outputs/lapse_model.joblib')

# Reload the saved model
loaded_model = joblib.load('outputs/lapse_model.joblib')

# Generate predictions using both models
original_predictions = selected_model.predict(X_test)
loaded_predictions = loaded_model.predict(X_test)

# Check whether predictions are identical
print("Predictions identical:",
      np.array_equal(original_predictions, loaded_predictions))

# 1. Get test customer IDs
test_customers = customer_features.loc[
    X_test.index, ['customer_id', 'lapsed']
].copy()

# 2. Calculate lapse probabilities using the saved model
test_customers['lapse_probability'] = (
    loaded_model.predict_proba(X_test)[:, 1]
)

# 3. Get Premium customers from your RFM segmentation
premium_customers = rfm.loc[
    rfm['is_premium'], ['customer_id']
]

# 4. Keep only Premium customers from the test set
premium_test = test_customers.merge(
    premium_customers,
    on='customer_id',
    how='inner'
)

# 5. Rank by highest predicted lapse probability
retention_list = (
    premium_test.sort_values(
        by='lapse_probability',
        ascending=False
    )
    .head(10)
    [['customer_id', 'lapse_probability', 'lapsed']]
    .rename(columns={'lapsed': 'actually_lapsed'})
)

# 6. Export CSV
retention_list.to_csv(
    'outputs/retention_list.csv',
    index=False
)

# 7. Verify
print(retention_list)

print("Premium customers in test set:", len(premium_test))
print("Customers in retention list:", len(retention_list))
print("Actually lapsed:", retention_list['actually_lapsed'].sum())