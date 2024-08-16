#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import ttest_ind

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

#%%
# Load your DataFrame
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/processed_data.csv')
food_df.head()

#%%
# Check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)

#%%
# Replace NaN values in eng_met with 0
food_df['eng_met'] = np.nan_to_num(food_df['eng_met'], nan=0)

#%%
#let's drop unnecessary columns
cols_drop = ['node_comments_disabled', 'node_typename', 'node_id', 'raw_caption',
     'timestamp','node_edge_media_preview_like_count',
       'node_owner_id', 'node_thumbnail_src', 'node_thumbnail_resources_0_src',
       'node_thumbnail_resources_0_config_width',
       'node_thumbnail_resources_0_config_height',
       'node_thumbnail_resources_1_src',
       'node_thumbnail_resources_1_config_width',
       'node_thumbnail_resources_1_config_height',
       'node_thumbnail_resources_2_src',
       'node_thumbnail_resources_2_config_width',
       'node_thumbnail_resources_2_config_height',
       'node_thumbnail_resources_3_src',
       'node_thumbnail_resources_3_config_width',
       'node_thumbnail_resources_3_config_height',
       'node_thumbnail_resources_4_src',
       'node_thumbnail_resources_4_config_width',
       'node_thumbnail_resources_4_config_height', 'node_is_video',
       'node_accessibility_caption', 'node_product_type',
       'node_video_view_count', 'hashtags', 'caption','sharpn']

food_df = food_df.drop(columns = cols_drop)

# %%
# Select specific columns for correlation, if needed
num_f = food_df.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", num_f)

col_int = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met'] 


correlation_matrix = food_df[col_int].corr()

# Create a colormap from the 'mako' color palette
mako_cmap = sns.color_palette("Blues", as_cmap=True)

# Plotting the heatmap with the 'mako' colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap with Mako Colormap')
plt.show()


#%%

#let's check for multicollinearity and significant variables
# Calculate VIF scores for each feature
X = food_df[col_int].dropna()

# Add a constant column for statsmodels
X = sm.add_constant(X)

# Create a DataFrame to store VIF scores
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Drop the constant column VIF score
vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

high_vif_features = vif_data[vif_data['VIF'] > 10]
print(high_vif_features)

#%%
#feature engineering
#color names- one hot encoding
# Select the specified columns
selected_df = food_df.copy()

# One-hot encode 'caption_lang' using 1 and 0
# caption_lang_encoded = pd.get_dummies(selected_df['caption_lang'], prefix='lang').astype(int)

# Convert the list of colors in 'color_names' to individual columns for one-hot encoding
color_names_exploded = selected_df['color_names'].apply(lambda x: x.strip("[]").replace("'", "").split(', ') if pd.notna(x) else [])

# Get only the top 3 dominant colors
top_colors_exploded = color_names_exploded.apply(lambda x: x[:3])

# Collect unique colors from the top 3
unique_colors = set(color for sublist in top_colors_exploded for color in sublist)

# Convert the set of unique colors to a sorted list for DataFrame columns
unique_colors_list = sorted(list(unique_colors))

# Initialize DataFrame for color one-hot encoding
color_one_hot = pd.DataFrame(0, index=selected_df.index, columns=unique_colors_list)

# Populate the color one-hot encoding DataFrame
for i, colors in enumerate(top_colors_exploded):
    for color in colors:
        if color in color_one_hot.columns:
            color_one_hot.loc[i, color] = 1

# Concatenate the one-hot encoded columns with the selected DataFrame

food_df = pd.concat([selected_df
# , caption_lang_encoded
, color_one_hot], axis=1).drop([ 'color_names'
                                # ,'caption_lang'
                                , 'dominant_colors'], axis=1)

# Display the first few rows of the resulting DataFrame
food_df.head()

#%%
# -------------------------------------------------------------------------
food_df.to_csv("/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv")

model_df = food_df.copy()

# %%
outlier_df = model_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met']]
outlier_df.shape

#%%

# Initialize a DataFrame to keep track of rows to remove
rows_to_remove = pd.Series([False] * len(model_df))

# Dictionary to keep track of the number of outliers per column
outlier_counts = {}

# Iterate through each numeric column to detect outliers
numeric_columns = outlier_df.select_dtypes(include=['float64', 'int64']).columns

for column_name in numeric_columns:
    # Calculate Q1, Q3, and IQR for outlier detection
    Q1 = outlier_df[column_name].quantile(0.25)
    Q3 = outlier_df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    is_outlier = (outlier_df[column_name] < (Q1 - 1.5 * IQR)) | (outlier_df[column_name] > (Q3 + 1.5 * IQR))
    
    # Count outliers
    num_outliers = is_outlier.sum()
    outlier_counts[column_name] = num_outliers

    # Update rows_to_remove series
    rows_to_remove = rows_to_remove | is_outlier

    # Create a box plot for visualization
    plt.figure(figsize=(10, 6))
    plt.boxplot(outlier_df[column_name], vert=False)
    plt.title(f'Box plot for {column_name}')
    plt.xlabel(column_name)
    plt.show()

# Remove the rows with any outliers
model_df = model_df[~rows_to_remove]

# Display the number of outliers per column
print("Number of outliers per column:")
for column_name, count in outlier_counts.items():
    print(f"{column_name}: {count}")

# Display the shape of the resulting DataFrame
print(f"Shape of DataFrame after removing outliers: {model_df.shape}")

#%%
#DATA SAMPLING - bootstrap sampling
# Set the target number of rows
target_rows = 1000

# Perform bootstrap sampling
model_df = model_df.sample(n=target_rows, replace=True, random_state=1)

# Save the new dataset to a CSV file
# bootstrap_sampled_data.to_csv('expanded_model_data.csv', index=False)

# Verify the shape of the new dataset
print("Shape of augmented dataset:", model_df.shape)

#%%
#save data in csv
model_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/bootstrapped_df.csv', index=False)

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
# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Fit the model on the training data
dt_regressor.fit(X_train_pca, y_train)

# Predict on the test data
y_pred_dt = dt_regressor.predict(X_test_pca)

# Calculate evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print the results
print('Decision Tree Regression results:')
print("Mean Squared Error:", mse_dt)
print("Mean Absolute Error:", mae_dt)
print("R-squared:", r2_dt)

#%%
# Random Forest 
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
# #MLP
# # Example feature dimensions
# input_dim = X_train_pca.shape[1]  # Number of features

# # Create an MLP model
# model = Sequential([
#     # Input layer
#     Dense(128, activation='relu', input_shape=(input_dim,)),
#     Dropout(0.2),  # Dropout layer to prevent overfitting
    
#     # Hidden layers
#     Dense(64, activation='relu'),
#     Dropout(0.2),
    
#     Dense(32, activation='relu'),
#     Dropout(0.2),
    
#     # Output layer
#     Dense(1)  # Output layer for regression
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='mean_squared_error',
#               metrics=['mean_absolute_error'])

# # Train the model
# history = model.fit(X_train_pca, y_train, 
#                     epochs=500,  # Adjust the number of epochs as needed
#                     batch_size=32,  # Adjust batch size as needed
#                     validation_data=(X_test_pca, y_test))

# # Evaluate the model
# y_pred_mlp = model.predict(X_test_pca)

# # Calculate evaluation metrics
# mse_mlp = mean_squared_error(y_test, y_pred_mlp)
# mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
# r2_mlp = r2_score(y_test, y_pred_mlp)

# # Print the results
# print('MLP Regression results:')
# print("Mean Squared Error:", mse_mlp)
# print("Mean Absolute Error:", mae_mlp)
# print("R-squared:", r2_mlp)

# %%
