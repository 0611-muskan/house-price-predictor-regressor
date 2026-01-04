import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing

# Import house dataset from sklearn
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['Price'] = housing.target
#top 5 rows
print(data.head())
#column with data types and non-null values
print(data.info())
#column with no. of non-null values 
# print(data.isnull().sum())

# No missing values in sklearn dataset, all data is numeric
         
# print(data.isnull().sum())
# print(data.info())

#handling duplicates
print(data.duplicated().sum())
data.drop_duplicates(inplace=True)
print(data.duplicated().sum())

#handling outliers 
print(data.describe())
#from matplotlib import pyplot as plt  
# data.boxplot(column='Price (in rupees)')
# plt.title('Boxplot of House Prices')
# plt.ylabel('Price (in rupees)')
# plt.show()

#using z-score to remove outliers
import numpy as np
from scipy import stats
# Select only numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
# Compute Z-scores for numeric columns
# Filter rows where all Z-scores are below 3
z_scores = np.abs(stats.zscore(data[numeric_cols], nan_policy='omit'))
data = data[(z_scores < 3).all(axis=1)]
data.columns = data.columns.str.strip()  # remove extra spaces
print("After removing outliers:")
print(data.describe())
print("Remaining rows:", len(data))
print("Numeric Columns:", list(numeric_cols))
print(data.info())

#label encoding - not needed for sklearn dataset (all numeric)
# Skip label encoding since all features are already numeric

# for col in data.columns:
#     print(f"{col} → {data[col].unique()[:10]}")


# Identify feature columns and target column
X = data.drop('Price', axis=1)
y = data['Price']

# Split into train-test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
# Save the trained model
import joblib       
joblib.dump(rf, 'house_price_model.pkl')

# Save the processed data (optional)