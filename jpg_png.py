#%%

#%%
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
import cv2

from sklearn.cluster import KMeans

#%%
# Load your DataFrame
sampled_images_df = pd.read_csv('sampl_food.csv')
sampled_images_df.head()

#%%

# Function to download an image from a URL
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

#%%
# Function to calculate sharpness
def calculate_sharpness(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

#%%
# Function to calculate colorfulness
def calculate_colorfulness(image):
    image = np.array(image)
    (B, G, R) = cv2.split(image.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = std_rg + std_yb + 0.3 * (mean_rg + mean_yb)
    return colorfulness

#%%
# Function to calculate the no. of colors
def calculate_number_of_colors(image, k=5):
    image = np.array(image)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    return len(clt.cluster_centers_)

#%%
# Function to calculate tone
def calculate_tone(image):
    image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    hue = image_hsv[:, :, 0]
    mean_hue = np.mean(hue)
    if mean_hue < 90:
        return "Cool"
    else:
        return "Warm"

#%%
# Function to estimate depth 
# def estimate_depth(image):
#     # Load MiDaS model
#     model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#     transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
#     model.eval()
    
#     # Convert image to torch.Tensor and apply transformations
#     img_tensor = transform(T.ToPILImage()(image)).unsqueeze(0)
    
#     # Compute depth
#     with torch.no_grad():
#         depth = model(img_tensor)
    
#     return depth.squeeze().numpy().mean()

def estimate_depth(image):
    midas = load_midas_model()
    input_batch = preprocess_midas(image)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    return depth_map

#%%
# Function to calculate clarity
def calculate_clarity(image):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    clarity_score = np.var(sharpness_map)
    return clarity_score

#%%
# Placeholder function for garnishing detection (using pre-trained object detection model)
def detect_garnishing(image):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    
    garnishing_label_id = 52  # You need to replace this with the actual label ID for garnishing
    
    garnishing_count = sum(1 for pred in predictions[0]['labels'] if pred == garnishing_label_id)
    
    return garnishing_count

# Placeholder function for portion size estimation
def estimate_portion_size(image):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)
    with torch.no_grad():
        predictions = model([image_tensor])
    
    if len(predictions[0]['boxes']) > 0:
        main_object_area = max([(box[2] - box[0]) * (box[3] - box[1]) for box in predictions[0]['boxes']])
        portion_size = main_object_area / (image.width * image.height)
    else:
        main_object_area = 0
    
    return portion_size.item()

#%%
# List to store the results as dictionaries
results_list = []

# Process each image
for idx, row in sampled_images_df.iterrows():
    image_num = row['image_num']
    image_tag = row['image_tag']
    image_url = row['image_url']
    
    image = download_image(image_url)
    if image:
        sharpness = calculate_sharpness(image)
        colorfulness = calculate_colorfulness(image)
        number_of_colors = calculate_number_of_colors(image)
        tone = calculate_tone(image)
        garnishing = detect_garnishing(image)
        portion_size = estimate_portion_size(image)
        depth = estimate_depth(image)
        clarity = calculate_clarity(image)
        
        # Append results as dictionary to results_list
        results_list.append({
            'image_num': image_num,
            'sharpness': sharpness,
            'colorfulness': colorfulness,
            'number_of_colors': number_of_colors,
            'tone': tone,
            'garnishing': garnishing,
            'portion_size': portion_size,
            'depth': depth,
            'clarity': clarity
        })
    else:
        print(f"Skipping image {image_num} due to download error.")

#%%
# Convert results_list to DataFrame
results_df = pd.DataFrame(results_list)

# Merge the results with the original dataset
final_df = pd.merge(sampled_images_df, results_df, on='image_num')

# Save the final DataFrame to a new CSV file
# final_df.to_csv('food_images_analysis_with_original.csv', index=False)
# print("Analysis complete. Results saved to 'food_images_analysis_with_original.csv'.")

# %%
final_df.head()

#%%