#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import os

from statsmodels.stats.outliers_influence import variance_inflation_factor

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#plots
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

#%%
#read model data
#without outlier/sampling
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv')

#without outliers
model_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_df.csv')

#whole bootstrapping sampling
bootstrapped_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/bootstrapped_df.csv')

#class imbalance
model_df_bal = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_df_balanced.csv')


#%%
#------------------------------------------------------------------------
np.random.seed(42)  
tf.random.set_seed(42)

#%%
#standardizing numerical columns
X = model_df_bal[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = model_df_bal['eng_met']

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
# B = X[numerical_features].dropna()

# # Add a constant column for statsmodels
# Z = sm.add_constant(B)

# # Create a DataFrame to store VIF scores
# vif_data = pd.DataFrame()
# vif_data['Feature'] = Z.columns
# vif_data['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]

# # Drop the constant column VIF score
# vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

# vif_data[vif_data['VIF'] > 10]['Feature']


#%%
# #PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :20]

# %%
#SPLITTIN
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# %%
# Initialize the linear regression model
lin_reg = LinearRegression()

# Train the model
lin_reg.fit(X_train_pca, y_train)

# Make predictions
y_pred_lr = lin_reg.predict(X_test_pca)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('Multiple Linear Regression results:')
print("Mean Squared Error:", mse_lr)
print("RMSE:", rmse_lr)
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

#%%
svr = SVR(kernel='rbf'
        #   , gamma = 0.15
          )
svr.fit(X_train_pca, y_train)

y_pred_svr = svr.predict(X_test_pca)

mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('Support Vector Regression results:')
print("Mean Squared Error:", mse_svr)
print("RMSE:", rmse_svr)
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)


#%%
# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Fit the model on the training data
dt_regressor.fit(X_train_pca, y_train)

# Predict on the test data
y_pred_dt = dt_regressor.predict(X_test_pca)

# Calculate evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print the results
print('Decision Tree Regression results:')
print("Mean Squared Error:", mse_dt)
print("RMSE:", rmse_dt)
print("Mean Absolute Error:", mae_dt)
print("R-squared:", r2_dt)

#%%
# Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

y_pred_rf = rf.predict(X_test_pca)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest results:')
print("Mean Squared Error:", mse_rf)
print("RMSE:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared:", r2_rf)

#%%
# Initialize the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred_xgb = model.predict(X_test_pca)

# Evaluate the model using regression metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print('XGBoost results:')
print("Mean Squared Error:", mse_xgb)
print("RMSE:", rmse_xgb)
print("Mean Absolute Error:", mae_xgb)
print("R-squared:", r2_xgb)

# %%
# Set seed for TensorFlow
tf.random.set_seed(42)
# Set GPU deterministic behavior
tf.config.experimental.enable_op_determinism()

# feature dimensions
input_dim = X_train_pca.shape[1]  

# Create an MLP model
model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2), 
    
    # Hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer
    Dense(1) 
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01),  
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# model.compile(optimizer='adam',
#               loss='mean_squared_error',
#               metrics=['mean_absolute_error'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_pca, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_test_pca, y_test),
                    callbacks=[early_stopping])


# Train the model
# history = model.fit(X_train_pca, y_train, 
#                     epochs=200,  # Adjust the number of epochs as needed
#                     batch_size=32,  # Adjust batch size as needed
#                     validation_data=(X_test_pca, y_test))

# Evaluate the model
y_pred_mlp = model.predict(X_test_pca)
# Reshape 
y_pred_mlp_f = y_pred_mlp.flatten()


# Calculate evaluation metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# Print the results
print('MLP Regression results:')
print("Mean Squared Error:", mse_mlp)
print("RMSE:", rmse_mlp)
print("Mean Absolute Error:", mae_mlp)
print("R-squared:", r2_mlp)

# %%

#--------------------------------------------------------------------------------

#%%
#plots
# Create a DataFrame for residuals
residuals_df = pd.DataFrame({
    'Linear Regression': y_test - y_pred_lr,
    'SVR': y_test - y_pred_svr,
    'Decision Tree': y_test - y_pred_dt,
    'Random Forest': y_test - y_pred_rf,
    'XGBoost': y_test - y_pred_xgb,
    'MLP': y_test - y_pred_mlp_f
})

# Custom color palette using the provided hex codes
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

# Plot Residuals Distribution
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.boxplot(data=residuals_df, palette=custom_palette)

plt.title('Residuals the for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('Residuals', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')


# Customize the appearance of the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

plt.show()


#---------------------------------------------------------------------------------
# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP'],
    'R^2': [r2_lr, r2_svr, r2_dt, r2_rf, r2_xgb, r2_mlp],
    'RMSE': [rmse_lr, rmse_svr, rmse_dt, rmse_rf, rmse_xgb, rmse_mlp]
})

#----------------------------------------------------------------------------
# Plot R^2 Values
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the barplot with a custom palette
barplot = sns.barplot(x='Model', y='R^2', data=metrics_df, palette=custom_palette, edgecolor="black")

# Add title and labels with enhanced formatting
plt.title('R^2 Scores for Different Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('R^2 Score', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

# Adjust the y-axis limits to zoom in on the R^2 values
# plt.ylim(-1, 0.5)  

# Add gridlines with custom styling
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Customize the spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

# Annotate bars with the R^2 values and apply text effects
for index, value in enumerate(metrics_df['R^2']):
    text = plt.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Add shadow effect to the text
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

plt.show()

#----------------------------------------------------------------------------

#rmse
# Custom color palette
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

# Create the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the barplot with a custom palette and edge colors
barplot = sns.barplot(x='Model', y='RMSE', data=metrics_df, palette=custom_palette, edgecolor="black")

# Add title and labels with enhanced formatting
plt.title('RMSE for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('RMSE', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

# Adjust the y-axis limits
min_rmse = min(metrics_df['RMSE']) - 0.2
max_rmse = max(metrics_df['RMSE']) + 0.2
plt.ylim(min_rmse, max_rmse)

# Add gridlines with custom styling
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Customize the spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

# Annotate bars with the RMSE values and apply text effects
for index, value in enumerate(metrics_df['RMSE']):
    text = plt.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Add shadow effect to the text
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

# Display the plot
plt.show()

#%%
# model_df = food_df.copy()

# # %%
# outlier_df = model_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met']]
# outlier_df.shape

# #%%

# # Initialize a DataFrame to keep track of rows to remove
# rows_to_remove = pd.Series([False] * len(model_df))

# # Dictionary to keep track of the number of outliers per column
# outlier_counts = {}

# # Iterate through each numeric column to detect outliers
# numeric_columns = outlier_df.select_dtypes(include=['float64', 'int64']).columns

# for column_name in numeric_columns:
#     # Calculate Q1, Q3, and IQR for outlier detection
#     Q1 = outlier_df[column_name].quantile(0.25)
#     Q3 = outlier_df[column_name].quantile(0.75)
#     IQR = Q3 - Q1

#     # Identify outliers
#     is_outlier = (outlier_df[column_name] < (Q1 - 1.5 * IQR)) | (outlier_df[column_name] > (Q3 + 1.5 * IQR))
    
#     # Count outliers
#     num_outliers = is_outlier.sum()
#     outlier_counts[column_name] = num_outliers

#     # Update rows_to_remove series
#     rows_to_remove = rows_to_remove | is_outlier

#     # Create a box plot for visualization
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(outlier_df[column_name], vert=False)
#     plt.title(f'Box plot for {column_name}')
#     plt.xlabel(column_name)
#     plt.show()

# # Remove the rows with any outliers
# model_df = model_df[~rows_to_remove]

# # Display the number of outliers per column
# print("Number of outliers per column:")
# for column_name, count in outlier_counts.items():
#     print(f"{column_name}: {count}")

# # Display the shape of the resulting DataFrame
# print(f"Shape of DataFrame after removing outliers: {model_df.shape}")

#%%
#--------------------------------------------------------------

#bootstrapping on whole data after removing outliers from target variable as well


#%%
#standardizing numerical columns
# model_df['eng_met'] = model_df['eng_met'].replace([np.inf, -np.inf], 0)

X = bootstrapped_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = bootstrapped_df['eng_met']

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
# B = X[numerical_features].dropna()

# # Add a constant column for statsmodels
# Z = sm.add_constant(B)

# # Create a DataFrame to store VIF scores
# vif_data = pd.DataFrame()
# vif_data['Feature'] = Z.columns
# vif_data['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]

# # Drop the constant column VIF score
# vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

# vif_data[vif_data['VIF'] > 10]['Feature']


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
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('Multiple Linear Regression results:')
print("Mean Squared Error:", mse_lr)
print("RMSE:", rmse_lr)
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

#%%
svr = SVR(kernel='rbf'
          , gamma = 0.15
         )
svr.fit(X_train_pca, y_train)

y_pred_svr = svr.predict(X_test_pca)

mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('Support Vector Regression results:')
print("Mean Squared Error:", mse_svr)
print("RMSE:", rmse_svr)
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)



#%%
# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Fit the model on the training data
dt_regressor.fit(X_train_pca, y_train)

# Predict on the test data
y_pred_dt = dt_regressor.predict(X_test_pca)

# Calculate evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print the results
print('Decision Tree Regression results:')
print("Mean Squared Error:", mse_dt)
print("RMSE:", rmse_dt)
print("Mean Absolute Error:", mae_dt)
print("R-squared:", r2_dt)

#%%
# Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

y_pred_rf = rf.predict(X_test_pca)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest results:')
print("Mean Squared Error:", mse_rf)
print("RMSE:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared:", r2_rf)

#%%
# Initialize the XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train_pca, y_train)

# Make predictions
y_pred_xgb = model.predict(X_test_pca)

# Evaluate the model using regression metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print('XGBoost results:')
print("Mean Squared Error:", mse_xgb)
print("RMSE:", rmse_xgb)
print("Mean Absolute Error:", mae_xgb)
print("R-squared:", r2_xgb)

#%%
#CNN
# # Example image dimensions
# img_height, img_width = 128, 128  # Adjust based on your images
# num_channels = 3  # RGB images

# # Create a CNN model
# model = Sequential([
#     # Convolutional layer
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     # Flatten the output from the convolutional layers
#     Flatten(),
    
#     # Fully connected layers
#     Dense(128, activation='relu'),
#     Dense(1)  # Output layer for regression
# ])

# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='mean_squared_error',
#               metrics=['mean_absolute_error'])

# # Train the model
# history = model.fit(X_train_pca, y_train, 
#                     epochs=10,  # Adjust number of epochs as needed
#                     batch_size=32,  # Adjust batch size as needed
#                     validation_data=(X_test_pca, y_test))

# # Evaluate the model
# y_pred_cnn = model.predict(X_test_pca)

# # Calculate evaluation metrics
# mse_cnn = mean_squared_error(y_test, y_pred_cnn)
# mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
# r2_cnn = r2_score(y_test, y_pred_cnn)

# # Print the results
# print('CNN Regression results:')
# print("Mean Squared Error:", mse_cnn)
# print("Mean Absolute Error:", mae_cnn)
# print("R-squared:", r2_cnn)

# %%
#MLP
# Example feature dimensions
input_dim = X_train_pca.shape[1]  # Number of features

# Create an MLP model
model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    
    # Hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Train the model
history = model.fit(X_train_pca, y_train, 
                    epochs=500,  # Adjust the number of epochs as needed
                    batch_size=32,  # Adjust batch size as needed
                    validation_data=(X_test_pca, y_test))

# Evaluate the model
y_pred_mlp = model.predict(X_test_pca)
y_pred_mlp_f = y_pred_mlp.flatten()

# Calculate evaluation metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

# Print the results
print('MLP Regression results:')
print("Mean Squared Error:", mse_mlp)
print("RMSE:", rmse_mlp)
print("Mean Absolute Error:", mae_mlp)
print("R-squared:", r2_mlp)


#%%
#plots
# Create a DataFrame for residuals
residuals_df = pd.DataFrame({
    'Linear Regression': y_test - y_pred_lr,
    'SVR': y_test - y_pred_svr,
    'Decision Tree': y_test - y_pred_dt,
    'Random Forest': y_test - y_pred_rf,
    'XGBoost': y_test - y_pred_xgb,
    'MLP': y_test - y_pred_mlp_f
})

# Custom color palette using the provided hex codes
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

# Plot Residuals Distribution
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.boxplot(data=residuals_df, palette=custom_palette)

plt.title('Residuals Distribution for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('Residuals', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')


# Customize the appearance of the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

plt.show()


#---------------------------------------------------------------------------------
# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP'],
    'R^2': [r2_lr, r2_svr, r2_dt, r2_rf, r2_xgb, r2_mlp],
    'RMSE': [rmse_lr, rmse_svr, rmse_dt, rmse_rf, rmse_xgb, rmse_mlp]
})

#----------------------------------------------------------------------------
#R^2
# Plot R^2 Values
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the barplot with a custom palette
barplot = sns.barplot(x='Model', y='R^2', data=metrics_df, palette=custom_palette, edgecolor="black")

# Add title and labels with enhanced formatting
plt.title('R^2 Scores for Different Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('R^2 Score', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

# Add gridlines with custom styling
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Customize the spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

# Annotate bars with the R^2 values and apply text effects
for index, value in enumerate(metrics_df['R^2']):
    text = plt.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Add shadow effect to the text
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

plt.show()

#----------------------------------------------------------------------------

#rmse
# Custom color palette
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

# Create the plot
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

# Create the barplot with a custom palette and edge colors
barplot = sns.barplot(x='Model', y='RMSE', data=metrics_df, palette=custom_palette, edgecolor="black")

# Add title and labels with enhanced formatting
plt.title('RMSE for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('RMSE', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

# Adjust the y-axis limits
min_rmse = min(metrics_df['RMSE']) - 0.2
max_rmse = max(metrics_df['RMSE']) + 0.2
plt.ylim(min_rmse, max_rmse)

# Add gridlines with custom styling
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

# Customize the spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

# Annotate bars with the RMSE values and apply text effects
for index, value in enumerate(metrics_df['RMSE']):
    text = plt.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    
    # Add shadow effect to the text
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

# Display the plot
plt.show()

#%%