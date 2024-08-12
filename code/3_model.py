#%%
#loading libraries
#basic
import pandas as pd
import numpy as np

#standardization
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#splitting data
from sklearn.model_selection import train_test_split

#models
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#mlp
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

#%%
#read model data
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_data.csv')
food_df.head()

#%%
#standardizing numerical columns

# model_df['eng_met'] = model_df['eng_met'].replace([np.inf, -np.inf], 0)

X = model_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = model_df['eng_met']

# Identify numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", numerical_features)

#%%
#standardizing
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X[numerical_features])
X_normalized = pd.DataFrame(X_normalized, columns=numerical_features)

X_normalized.head()

#%%
B = X[numerical_features].dropna()

# Add a constant column for statsmodels
Z = sm.add_constant(B)

# Create a DataFrame to store VIF scores
vif_data = pd.DataFrame()
vif_data['Feature'] = Z.columns
vif_data['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]

# Drop the constant column VIF score
vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

vif_data[vif_data['VIF'] > 10]['Feature']


#%%
# #PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :20]

# %%
#SPLITTIN
# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# %%
# Initialize the linear regression model
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X_train_pca, y_train)

# Make predictions
y_pred_lr = lin_reg.predict(X_test_pca)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('Multiple Linear Regression results:')
print("Mean Squared Error:", mse_lr)
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

#%%
# Initialize the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred = model.predict(X_test_pca)

# Evaluate the model using regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('XGBoost results:')
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

#%%

gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_pca, y_train)

y_pred_gbr = gbr.predict(X_test_pca)

mse_gbr = mean_squared_error(y_test, y_pred_gbr)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

print('Gradient Boosting results:')
print("Mean Squared Error:", mse_gbr)
print("Mean Absolute Error:", mae_gbr)
print("R-squared:", r2_gbr)

#%%

from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(X_train_pca, y_train)

y_pred_svr = svr.predict(X_test_pca)

mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('Support Vector Regression results:')
print("Mean Squared Error:", mse_svr)
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)

#%%

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

y_pred_rf = rf.predict(X_test_pca)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest results:')
print("Mean Squared Error:", mse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared:", r2_rf)

#%%

# Train the MLP model on the cleaned data
mlp_cleaned = MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu', max_iter=1000, random_state=42)
mlp_cleaned.fit(X_train_pca, y_train)

# Make predictions
y_pred_cleaned = mlp_cleaned.predict(X_test_pca)

# Evaluate the model
mse_cleaned = mean_squared_error(y_test, y_pred_cleaned)
r2_cleaned = r2_score(y_test, y_pred_cleaned)

print('MLP results:')
print(f"Mean Squared Error: {mse_cleaned}")
print(f"R-squared: {r2_cleaned}")




#%%

#%%