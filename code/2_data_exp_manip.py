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

#xgboost
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#%%
# Load your DataFrame
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/eng_met.csv')
food_df.head()

#%%
# Check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)

#%%
# Replace NaN values in eng_met with 0
food_df['eng_met'] = np.nan_to_num(food_df['eng_met'], nan=0)

#%%
#unique values in some particular cols

# %%
# Select specific columns for correlation, if needed
col_int = ['sharpness', 'colorfulness', 'depth',  'hue', 'saturation', 'brightness', 'dim_w', 'dim_h', 'rule_of_thirds_x', 'rule_of_thirds_y', 'symmetry_score', 'tone', 'center_score', 'mean_rgb', 'lines_count'] 


correlation_matrix = food_df[col_int].corr()

# Create a colormap from the 'mako' color palette
mako_cmap = sns.color_palette("Blues", as_cmap=True)

# Plotting the heatmap with the 'mako' colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap with Mako Colormap')
plt.show()

#remove rule or third or dimension?

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

vif_data[vif_data['VIF'] > 10]['Feature']

#%%
#find outliers
model_df = food_df.copy()

#%%
#feature engineering
#color names- one hot encoding
# Select the specified columns
selected_df = model_df.copy()

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

model_df = pd.concat([selected_df
# , caption_lang_encoded
, color_one_hot], axis=1).drop([ 'color_names'
                                # ,'caption_lang'
                                , 'dominant_colors'], axis=1)

# Display the first few rows of the resulting DataFrame
model_df.head()

# %%
outlier_df = food_df.drop(columns=['shortcode', 'timestamp', 'like_count', 'comment_count', 'hashtags', 'caption', 'display_url', 'number_of_colors', 'tone_cat', 'garnishing', 'clarity', 'dominant_colors', 'followers', 'pattern_score', 'caption_lang', 'color_names', 'exp_growth', 'rule_of_thirds_x', 'rule_of_thirds_y', 'mean_rgb', 'brightness'
                          ])
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
#save data in csv
# model_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_data.csv', index=False)

#%%
#standardizing numerical columns
# Data preprocessing pipeline
# Define features and target, drop categorical features

model_df['eng_met'] = model_df['eng_met'].replace([np.inf, -np.inf], 0)

X = model_df.drop(columns=['eng_met', 'shortcode', 'timestamp', 'like_count', 'comment_count', 'hashtags', 'caption', 'display_url', 'number_of_colors', 'tone_cat', 'garnishing', 'clarity',  'followers', 'pattern_score', 'caption_lang', 'exp_growth', 'rule_of_thirds_x', 'rule_of_thirds_y', 'mean_rgb', 'brightness'
                          ])


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
#PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :10]

# %%
#SPLITTIN
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

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

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# %%
svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_model.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = svr_model.predict(X_test_pca)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('SVR RMSE:', rmse)
print('SVR MAE:', mae)
print('SVR R-squared:', r2)

#%%

#%%
