#%%
# with the new data for checking the efficiency of the model. Timestamp 
# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models

#%%
# Load the data
data = pd.read_csv('/Users/rajnavalakha/Documents/Final Sem Project UCD/Mock Data/eng_met.csv')

numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].replace([np.inf, -np.inf], np.nan)

# Optionally, print the rows with NaNs to inspect
print("Rows with NaN values:\n", data[numerical_cols].isnull().sum())

# Fill or drop NaN values as appropriate
# For example, fill NaNs with the mean of the column
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

#%%
# Function to download and process images
def fetch_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((128, 128))  # Resize for consistency
        return np.array(img) / 255.0  # Normalize pixel values
    except:
        return None

#%%
# Apply the image processing function to the 'display_url' column
data['image_data'] = data['display_url'].apply(fetch_image)

# Drop rows where image fetching failed
data = data.dropna(subset=['image_data'])

# Prepare image data for the model
image_data = np.stack(data['image_data'].values)

#%%
# Define the MLP model
model = models.Sequential([
    layers.Flatten(input_shape=(128, 128, 3)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Adjust the output layer based on your prediction task
])

model.compile(optimizer='adam', loss='mean_squared_error')

#%%
# Split data into features and labels
X = image_data
y = data['eng_met']  # Replace 'target_column' with the actual column name

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mean_adjusted_error = np.sqrt(mse)

#%%
print('Mean Adjusted Error:', mean_adjusted_error)
print('Mean squared error', mse)
# %%
