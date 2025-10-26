TRAIN_PATH = '../Dataset/train.csv'
TEST_PATH = '../Dataset/test.csv'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Load and Preprocess Data ---
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train['HotelValue']
X = train.drop(['Id', 'HotelValue'], axis=1)

numeric_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

# Handle missing values
for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())
for col in categorical_features:
    X[col] = X[col].fillna(X[col].mode()[0])

# One-hot encode
X_processed = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Create train-test split for validation
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- Hyperparameter Grid ---
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# --- GridSearchCV ---
print("Starting GridSearchCV for Gradient Boosting...")
gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score (MSE): {-grid_search.best_score_:.2f}")

# --- Train Best Model ---
best_model = grid_search.best_estimator_
val_preds = best_model.predict(X_val)

# Evaluate
mae = mean_absolute_error(y_val, val_preds)
mse = mean_squared_error(y_val, val_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, val_preds)

print(f"\n--- Best Model Validation Metrics ---")
print(f"MAE: ${mae:,.2f}")
print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²: {r2:.4f}")

# --- Visualization ---
results = pd.DataFrame(grid_search.cv_results_)

# Plot 1: Top 20 Hyperparameter Combinations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sort by mean test score
results_sorted = results.sort_values('mean_test_score', ascending=False).head(20)
axes[0, 0].barh(range(20), -results_sorted['mean_test_score'])
axes[0, 0].set_xlabel('Mean Squared Error')
axes[0, 0].set_ylabel('Configuration Rank')
axes[0, 0].set_title('Top 20 Hyperparameter Configurations')
axes[0, 0].invert_yaxis()

# Plot 2: Learning Rate vs Performance
lr_groups = results.groupby('param_learning_rate')['mean_test_score'].mean()
axes[0, 1].plot(lr_groups.index.astype(float), -lr_groups.values, marker='o', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Learning Rate')
axes[0, 1].set_ylabel('Mean Squared Error')
axes[0, 1].set_title('Learning Rate vs Model Performance')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Max Depth vs Performance
depth_groups = results.groupby('param_max_depth')['mean_test_score'].mean()
axes[1, 0].bar(depth_groups.index.astype(int), -depth_groups.values, color='coral')
axes[1, 0].set_xlabel('Max Depth')
axes[1, 0].set_ylabel('Mean Squared Error')
axes[1, 0].set_title('Max Depth vs Model Performance')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: N_estimators vs Performance
est_groups = results.groupby('param_n_estimators')['mean_test_score'].mean()
axes[1, 1].plot(est_groups.index.astype(int), -est_groups.values, marker='s', linewidth=2, markersize=8, color='green')
axes[1, 1].set_xlabel('Number of Estimators')
axes[1, 1].set_ylabel('Mean Squared Error')
axes[1, 1].set_title('Number of Estimators vs Model Performance')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_boosting_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Predict on Test Set ---
test_ids = test['Id']
X_test = test.drop(['Id'], axis=1)

for col in numeric_features:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna(X[col].median())
for col in categorical_features:
    if col in X_test.columns:
        X_test[col] = X_test[col].fillna(X[col].mode()[0])

X_test_processed = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
X_test_processed = X_test_processed.reindex(columns=X_processed.columns, fill_value=0)

# Retrain on full training data
final_model = GradientBoostingRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_processed, y)
test_preds = final_model.predict(X_test_processed)

# Save predictions
gradient_df = pd.DataFrame({'Id': test_ids, 'HotelValue': test_preds})
gradient_df.to_csv('gradient_tuned.csv', index=False)
print("\ngradient_tuned.csv has been created!")
