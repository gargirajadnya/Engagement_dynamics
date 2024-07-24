#%%

#%%

#basic
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

#comp vision and AI
import requests
from io import BytesIO
from PIL import Image
import os
import torch
import torchvision.transforms as T
from torchvision import transforms
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from collections import Counter
from sklearn.cluster import KMeans

#%%
# Load your DataFrame
sampled_images_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/clean_data.csv')
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

#
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

#
# Function to calculate the no. of colors
def calculate_number_of_colors(image, k=5):
    image = np.array(image)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = KMeans(n_clusters=k)
    clt.fit(image)
    return len(clt.cluster_centers_)

#
# Function to calculate tone
def calculate_tone(image):
    image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    hue = image_hsv[:, :, 0]
    mean_hue = np.mean(hue)
    if mean_hue < 90:
        return "Cool"
    else:
        return "Warm"

#
# Function to estimate depth 
def load_midas_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    midas.eval()
    return midas


# Function to preprocess the image for MiDaS model
def preprocess_midas(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(image).unsqueeze(0)
    return input_batch

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
    
    depth_map = prediction.cpu().numpy().mean()
    return depth_map

#
# Function to calculate clarity
def calculate_clarity(image):
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness_map = cv2.Laplacian(gray_image, cv2.CV_64F)
    clarity_score = np.var(sharpness_map)
    return clarity_score

#
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

#
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


#
# Function to extract RGB values and calculate statistics
def extract_rgb_values(image):
    image_np = np.array(image)
    pixels = image_np.reshape((-1, 3))  # Reshape to a list of pixels
    
    # Calculate mean RGB values
    mean_rgb = np.mean(pixels, axis=0)
    
    # Calculate median RGB values
    median_rgb = np.median(pixels, axis=0)

    # Calculate HSV values
    image_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue = image_hsv[:,:,0]
    saturation = image_hsv[:,:,1]
    brightness = image_hsv[:,:,2]
    
    hue = np.mean(hue)
    saturation = np.mean(saturation)
    brightness = np.mean(brightness)
    
    # median_hue = np.median(hue)
    # median_saturation = np.median(saturation)
    # median_brightness = np.median(brightness)
    
    #return mean_rgb, median_rgb, mean_hue, median_hue, mean_saturation, median_saturation, mean_brightness, median_brightness

    
    # Use KMeans to find the dominant colors
    num_clusters = 5
    clt = KMeans(n_clusters=num_clusters)
    clt.fit(pixels)
    
    # Count the number of pixels assigned to each cluster
    counts = Counter(clt.labels_)
    
    # Get the dominant colors (sorted by frequency)
    dominant_colors = [clt.cluster_centers_[i] for i in counts.keys()]
    
    # Sort by frequency
    dominant_colors = sorted(dominant_colors, key=lambda x: counts[clt.predict([x])[0]], reverse=True)
    
    return mean_rgb, median_rgb, dominant_colors, hue, saturation, brightness

#
# Function to evaluate Rule of Thirds
def evaluate_rule_of_thirds(image):
    height, width, _ = image.shape
    thirds_y = height // 3
    thirds_x = width // 3
    return thirds_x, thirds_y

# Function to evaluate symmetry
def evaluate_symmetry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(gray, 1)
    difference = cv2.absdiff(gray, flipped)
    score = np.sum(difference)
    return score

# Function to evaluate lines (horizontal, vertical, diagonal)
def evaluate_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    horizontal, vertical, diagonal = 0, 0, 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)
            if angle < 10 or angle > 170:
                vertical += 1
            elif 80 < angle < 100:
                horizontal += 1
            else:
                diagonal += 1
    return horizontal, vertical, diagonal

# Function to evaluate patterns
def evaluate_patterns(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    if corners is not None:
        return len(corners)
    return 0

# Function to evaluate triangles
def evaluate_triangles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    triangles = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            triangles += 1
    return triangles

# Function to evaluate center composition
def evaluate_center_composition(image):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 10
    center_region = image[center_y - radius:center_y + radius, center_x - radius:center_x + radius]
    return np.mean(center_region)


#%%
# List to store the results as dictionaries
results_list = []

# Process each image
for idx, row in sampled_images_df.iterrows():
    image_node = row['node_shortcode']
    image_com = row['node_edge_media_to_comment_count']
    image_likes = row['node_edge_liked_by_count']
    image_caption = row['node_edge_media_to_caption_edges_0_node_text']
    image_dim_h = row['node_dimensions_height']
    image_dim_w = row['node_dimensions_width']
    image_url = row['node_display_url']
    
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
        #hue, saturation, brightness = extract_hsv_values(image)
        #Convert PIL image to OpenCV format for composition evaluation
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        thirds_x, thirds_y = evaluate_rule_of_thirds(image_cv)
        symmetry_score = evaluate_symmetry(image_cv)
        horizontal, vertical, diagonal = evaluate_lines(image_cv)
        pattern_score = evaluate_patterns(image_cv)
        triangle_count = evaluate_triangles(image_cv)
        center_score = evaluate_center_composition(image_cv)
        
        
        # Extract RGB values
        mean_rgb, median_rgb, dominant_colors, hue, saturation, brightness = extract_rgb_values(image)
        
        # Append results as dictionary to results_list
        results_list.append({
            'image_node': image_node,
            'sharpness': sharpness,
            'colorfulness': colorfulness,
            'number_of_colors': number_of_colors,
            'tone': tone,
            'garnishing': garnishing,
            'portion_size': portion_size,
            'depth': depth,
            'clarity': clarity,
            'mean_rgb': mean_rgb,
            #'median_rgb': median_rgb,
            'dominant_colors': dominant_colors,
            'hue': hue,
            'saturation': saturation,
            'brightness': brightness,
            'rule_of_thirds_x': thirds_x,
            'rule_of_thirds_y': thirds_y,
            'symmetry_score': symmetry_score,
            'lines_horizontal': horizontal,
            'lines_vertical': vertical,
            'lines_diagonal': diagonal,
            'pattern_score': pattern_score,
            'triangle_count': triangle_count,
            'center_score': center_score
        })
    else:
        print(f"Skipping image {image_node} due to download error.")

#%%
# Convert results_list to DataFrame
results_df = pd.DataFrame(results_list)

# Merge the results with the original dataset
final_df = pd.merge(sampled_images_df, results_df, left_on='node_shortcode', right_on='image_node')


# Save the final DataFrame to a new CSV file
# final_df.to_csv('food_images_analysis_with_original.csv', index=False)
# print("Analysis complete. Results saved to 'food_images_analysis_with_original.csv'.")

# %%
final_df.head()

#%%

#separate the bracket cols into individual cols
def split_rgb_list(df, col_name):
    # Split the RGB values into separate columns
    df[[f'{col_name}_r', f'{col_name}_g', f'{col_name}_b']] = pd.DataFrame(df[col_name].tolist(), index=df.index)
    return df

# Apply function to split mean_rgb column
df = split_rgb_list(final_df, 'mean_rgb')

df.head()
# %%
#df.to_csv('food_images_analysis_with_original.csv', index=False)

