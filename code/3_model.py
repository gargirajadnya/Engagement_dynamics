#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import os

#standardization
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#splitting data
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#mlp
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#%%
#read model data
#without outlier/sampling
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv')

#%%
#------------------------------------------------------------------------

#%%
col_int = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 
            'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met']

# Extract numerical features and target variable
X_num = food_df[col_int[:-1]]  # All columns except the target
y = food_df['eng_met']

#%%
# Standardize numerical features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

#%%
# Split the dataset
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num_scaled, y, test_size=0.2, random_state=42)

#%%
# Load pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

#%%
# Function to preprocess and extract features from a local image file
def extract_features(image_path, model):
    try:
        # Load the image
        img = image.load_img(image_path, target_size=(224, 224)) 
        
        # Convert the image to an array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extract features
        features = model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return np.zeros(model.output_shape[1]) 
    
#%%
# Extract features for all images
image_paths = [os.path.join("/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/code/saved_images", f"{shortcode}.jpg") for shortcode in food_df['shortcode']]

image_features = np.array([extract_features(image_path, base_model) for image_path in image_paths])

#%%
# Split image features into training and testing sets, ensuring the same indices as the numerical data
X_train_img, X_test_img = train_test_split(image_features, test_size=0.2, random_state=42)

#%%
# Combine numerical and image features for training and testing sets
X_train_combined = np.hstack((X_train_num, X_train_img))
X_test_combined = np.hstack((X_test_num, X_test_img))

#%%
# Define the MLP model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

#%%
# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_combined, y_train, epochs=10000, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

#%%
# Evaluate on the test set
loss = model.evaluate(X_test_combined, y_test)
print(f'Test Loss: {loss}')












# #%%
# #loading libraries
# #basic
# import pandas as pd
# import numpy as np

# #standardization
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler

# #splitting data
# from sklearn.model_selection import train_test_split

# #metrics
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# #mlp
# import requests
# from PIL import Image
# from io import BytesIO
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout

# #%%
# #read model data
# #without outlier/sampling
# food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv')
# food_df.head()

# #%%
# #------------------------------------------------------------------------

# #%%
# col_int = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 
#            'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met']

# # Extract numerical features and target variable
# X_num = food_df[col_int[:-1]] 
# y = food_df['eng_met']

# #%%
# # Standardize numerical features
# scaler = StandardScaler()
# X_num_scaled = scaler.fit_transform(X_num)

# #%%
# # Split the dataset
# X_train_num, X_test_num, y_train, y_test = train_test_split(X_num_scaled, y, test_size=0.2, random_state=42)

# #%%
# # Load pre-trained ResNet50 model for feature extraction
# base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# #%%
# # Function to preprocess and extract features from a saved image
# def extract_features(image_path, model):
#     try:
#         # Load the image
#         img = image.load_img(image_path, target_size=(224, 224)) 
        
#         # Convert the image to an array
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
        
#         # Extract features
#         features = model.predict(img_array)
#         return features.flatten()
#     except Exception as e:
#         print(f"Failed to process image {image_path}: {e}")
#         return np.zeros(model.output_shape[1]) 
    
# #%%
# # Extract features for all images
# image_paths = [os.path.join("/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/code/saved_images", f"{shortcode}.jpg") for shortcode in food_df['shortcode']]

# image_features = [extract_features(image_path, base_model) for image_path in image_paths]

# #%%
# # Split image features into training and testing sets, ensuring the same indices as the numerical data
# X_train_img, X_test_img = train_test_split(image_features, test_size=0.2, random_state=42)

# #%%
# # Combine numerical and image features for training and testing sets
# X_train_combined = np.hstack((X_train_num, X_train_img))
# X_test_combined = np.hstack((X_test_num, X_test_img))

# #%%
# # Define the MLP model
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],)),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(1)  # Output layer for regression
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')


# #%%
# # Train the model
# #history = model.fit(X_train_combined, y_train, epochs=10000, batch_size=32, validation_split=0.1)

# #%%
# # Evaluate on the test set
# loss = model.evaluate(X_test_combined, y_test)
# print(f'Test Loss: {loss}')

# #%%

# %%
