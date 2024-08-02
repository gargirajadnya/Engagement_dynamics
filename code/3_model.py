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

#splitting data
from sklearn.model_selection import train_test_split

#SVM
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#NEURAL NETWORK
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

#%%
#read model data
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_data.csv')
food_df.head()

#%%
#%%
#standardizing numerical columns
# Data preprocessing pipeline
# Define features and target, drop categorical features

#!!!!!!!!SHALL WE REMOVE MEANRGB COLS????!!!!!!!!!!

X = food_df.drop(columns=['eng_met', 'shortcode', 'timestamp', 'display_url', 'tone_cat', 'hashtags', 'garnishing','like_count', 'comment_count', 'followers', 'pattern_score'
                        #   ,'lines_horizontal',  'lines_diagonal', 'triangle_count'
                          ])


y = food_df['eng_met']

# Identify numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", numerical_features)

#%%
# Data preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features)
    ])

# Fit and transform the entire data
X_transformed = preprocessor.fit_transform(X)

# Manually rename the columns to remove the 'num_' prefix
transformed_column_names = numerical_features

# Convert the transformed data back to DataFrame for better readability
X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_column_names)


# Display the transformed data
X_transformed_df.head()

#%%
# Save the transformed data along with the target variable
transformed_data = pd.concat([X_transformed_df, food_df['eng_met'].reset_index(drop=True)], axis=1)

# Save to a new CSV file (optional)
# transformed_data.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/transformed_eng_met.csv', index=False)

# Display the transformed data with the target variable
transformed_data.head()

# %%
#SPLITTING
# Split the data into features and target
X_transformed = transformed_data.drop(columns=['eng_met'])
y = transformed_data['eng_met']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Display the shapes of the split data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# %%

# Create and train the SVR model
svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svr_model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('SVR RMSE:', rmse)
print('SVR MAE:', mae)
print('SVR R-squared:', r2)

#%%