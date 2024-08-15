#%%

#%%
#basic
import pandas as pd
import numpy as np

#preprocessing
import sklearn.preprocessing as pp
import sklearn.impute as imp
import re
import string
import os
import langid
from datetime import datetime
#color names
import webcolors
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import re
import ast

#comp vision and AI
import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import transforms
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
from collections import Counter
from sklearn.cluster import KMeans

#%%
directory_path = '/Users/rajnavalakha/Documents/Engagement_dynamics/data'

def sort_key(raw):
    return int(raw.split('_')[1].split('.')[0])

# Get a list of all CSV files in the directory and sort them using the custom key
csv_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.csv')], key=sort_key)

#%% Read and concatenate all CSV files
sampled_images_df = pd.concat([pd.read_csv(os.path.join(directory_path, file)) for file in csv_files], ignore_index=True)

#%%load follower data
follower_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/followers.csv')

#%%
#drop duplicates
sampled_images_df = sampled_images_df.drop_duplicates(subset='node/shortcode')
sampled_images_df.shape

#%%
# Merging both dataframes on shortcode
# sampled_images_df = pd.merge(sampled_images_df, follower_df, on='node/shortcode')
sampled_images_df.head()


#%%
#prepare the column names
def prep_colnames(my_df):
    # Make a copy of the dataframe
    my_df = pd.DataFrame(my_df).copy()

    # Get the column names and process them
    col_names = list(my_df.columns)
    col_names = [str(x).strip() for x in col_names]  # Strip whitespace
    col_names = [str(x).lower() for x in col_names]  # Lowercase

    # Replace punctuation/spaces with _
    str_lookup = "_" * len(string.punctuation + string.whitespace)
    trans_table = str.maketrans(string.punctuation + string.whitespace, str_lookup, "")
    col_names = [x.translate(trans_table) for x in col_names]

    # Remove trailing and leading "_"
    col_names = [x.strip("_") for x in col_names]

    # Remove multiple "_"
    col_names = [re.sub(pattern="_+", repl="_", string=x) for x in col_names]

    # Assign the processed column names back to the dataframe
    my_df.columns = col_names

    return my_df

#%%
# Function to extract terms beginning with "#"
def extract_hashtags(text):
    if not isinstance(text, str):
        return []  # Or handle this case as appropriate
    return re.findall(r'#\w+', text)

#%%
# Function to extract clean text
def clean_text(text):
    if not isinstance(text, str):
        return ''  # or handle it in another appropriate way
    # Example cleaning process (customize as needed)
    text = text.lower()  # convert to lowercase
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    text = text.strip()  # remove leading and trailing whitespace
    return text

def clean_hashtags(text):
    # Remove specific unwanted sequences
    text = re.sub(r'#ÿ™|#Âπ≥|#–∫–æ|#–∫|#–≤|#–Ω|#Îπµ', '', text)
    
    # Remove extra # that do not have text following them
    text = re.sub(r'#\s*', '#', text)
    
    # Remove any leading, trailing, or multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove multiple consecutive # characters
    text = re.sub(r'#+', '#', text)
    
    return text

# Function to fix encoding with error handling
def fix_encoding(text):
    return text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')

    
#%%
sampled_images_df = prep_colnames(sampled_images_df)
sampled_images_df.head()

#%% 
# Rename selected columns
rename_dict = {
    'node_shortcode': 'shortcode',
    'node_edge_liked_by_count': 'like_count',
    'node_edge_media_to_comment_count':'comment_count',
    'node_dimensions_height':'dim_h',
    'node_dimensions_width':'dim_w',
    'node_display_url':'display_url',
    'node_taken_at_timestamp':'timestamp',
    'node_edge_media_to_caption_edges_0_node_text':'raw_caption'
}
sampled_images_df.rename(columns=rename_dict, inplace=True)


#%%
# Apply the function to the 'text' column and create a new column with the extracted hashtags
sampled_images_df['hashtags'] = sampled_images_df['raw_caption'].apply(extract_hashtags)
sampled_images_df['caption'] = sampled_images_df['raw_caption'].apply(clean_text)
# sampled_images_df['hashtags'] = sampled_images_df['hashtags'].apply(fix_encoding)
# sampled_images_df['hashtags'] = sampled_images_df['hashtags'].apply(clean_hashtags)

#%%
#handle missing values
sampled_images_df['hashtags'] = sampled_images_df['hashtags'].replace('', 'NA')
sampled_images_df['caption'] = sampled_images_df['caption'].replace('', 'NA')

# Display the DataFrame
sampled_images_df.head()

#%%
# Convert columns to integers
sampled_images_df['like_count'] = sampled_images_df['like_count'].fillna(0).astype(int)
sampled_images_df['comment_count'] = sampled_images_df['comment_count'].fillna(0).astype(int)
# sampled_images_df['followers'] = sampled_images_df['followers'].fillna(0).astype(int)

#%%
# Convert timestamp to datetime
sampled_images_df.loc[:, 'timestamp'] = sampled_images_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

#%%
# Remove rows where followers == 999
# sampled_images_df = sampled_images_df[sampled_images_df['followers'] != 999]

#%%
#SAMPLING
# Filter the DataFrame to include only rows where the timestamp is earlier than '25/07/2024'
sampled_images_df = sampled_images_df[sampled_images_df['timestamp'] < '2024-08-10 00:00:00']
sampled_images_df.shape

#%%
#ENGAGEMENT METRICS

#creating engagement metric
sampled_images_df['eng_met'] = (sampled_images_df['like_count'] + (2 * sampled_images_df['comment_count'])).round(2)

sampled_images_df.head()

#%%
sampled_images_df_copy = sampled_images_df.copy()
sampled_images_df.shape

# %%
# sampled_images_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/clean_data.csv', index=False)

# %%
#--------------------------------------------------------------------------------------------------------------------------------
#%%
#Image recognition and aesthetics
# Function to download an image from a URL
# def download_image(url):
#     try:
#         response = requests.get(url, timeout=5)
#         response.raise_for_status()
#         return Image.open(BytesIO(response.content))
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to download {url}: {e}")
#         return None

#save images
#------
# Directory to save images
# save_directory = "saved_images"
# os.makedirs(save_directory, exist_ok=True)

# def download_and_save_image(image_url, image_name):
#     try:
#         response = requests.get(image_url)
#         response.raise_for_status()  # Check if the request was successful
        
#         # Convert the response content to an image
#         image = Image.open(BytesIO(response.content))
        
#         # Save the image to the specified directory
#         image_path = os.path.join(save_directory, f"{image_name}.jpg")
#         image.save(image_path)
        
#         print(f"Image saved at: {image_path}")
#         return image_path
#     except Exception as e:
#         print(f"Failed to download or save image: {e}")
#         return None

# # Example usage:
# for idx, row in sampled_images_df.iterrows():
#     shortcode = row['shortcode']
#     image_url = row['display_url']
    
#     image_path = download_and_save_image(image_url, shortcode)
#     if image_path:
#         print(f"Successfully saved image {shortcode}")


# Apply the download_image function to each URL in the 'display_url' column
# sampled_images_df_copy['image'] = sampled_images_df['display_url'].apply(download_image)

#%%
# call images
# def load_image(image_path):
#     image = Image.open(image_path)
#     # image = image.resize((224, 224))  
#     image_array = np.array(image) 
#     return image_array

# # Example usage:
# image_array = load_image("saved_images/example.jpg")
# Now you can use `image_array` for prediction in your model

#%%
# Function to detect the language
def detect_language(text):
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return 'na'


#%%
#FUNCTIONS
# Function to calculate sharpness
def calculate_sharpness(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance_of_laplacian

#another sharpness_2 function
def calculate_sharpn(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return sharpness


# Function to calculate exposure
def calculate_exposure(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    exposure = np.mean(gray_image)
    return exposure

# Function to calculate brilliance
def calculate_brilliance(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    brilliance = np.std(gray_image)
    return brilliance
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
# Function to calculate highlights
def calculate_highlights(image, threshold=200):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    highlights = np.sum(gray_image > threshold)
    return highlights

# Function to calculate vibrancy
def calculate_vibrancy(image):
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    saturation = hsv_image[:, :, 1]
    brightness = hsv_image[:, :, 2]
    vibrancy = np.mean(saturation * brightness)
    return vibrancy

# Function to calculate warmth
def calculate_warmth(image):
    hsv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    warmth = np.mean(hsv_image[:, :, 0])
    return warmth

# Function to calculate tint
def calculate_tint(image):
    lab_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2Lab)
    tint = np.mean(lab_image[:, :, 1] - lab_image[:, :, 2])
    return tint

# Function to calculate definition
def calculate_definition(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    definition = np.sum(edges)
    return definition

# Function to calculate noise_reduction
def calculate_noise_reduction(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    noise = np.std(gray_image)
    return noise

# Function to calculate vignette
def calculate_vignette(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Calculate the mean pixel values for each edge of the image
    top_edge_mean = np.mean(gray_image[0, :])
    bottom_edge_mean = np.mean(gray_image[-1, :])
    left_edge_mean = np.mean(gray_image[:, 0])
    right_edge_mean = np.mean(gray_image[:, -1])
    
    # Calculate the average of these means
    edge_mean = np.mean([top_edge_mean, bottom_edge_mean, left_edge_mean, right_edge_mean])
    
    # Calculate the center value
    center_value = gray_image[gray_image.shape[0] // 2, gray_image.shape[1] // 2]
    
    # Calculate vignette effect as the difference between center and edge mean values
    vignette = center_value - edge_mean
    
    return vignette


# Function to calculate tone
def calculate_tone(image):
    image_hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    hue = image_hsv[:, :, 0]
    mean_hue = np.mean(hue)
    if mean_hue < 90:
        return "Cool"
    else:
        return "Warm"

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
# Function to calculate shadows
def calculate_shadows(image, threshold=50):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    shadows = np.sum(gray_image < threshold)
    return shadows

# Function to calculate contrast
def calculate_contrast(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    contrast = np.max(gray_image) - np.min(gray_image)
    return contrast

# Function to calculate black_point
def calculate_black_point(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    black_point = np.min(gray_image)
    return black_point

#
# # Placeholder function for garnishing detection (using pre-trained object detection model)
# def detect_garnishing(image):
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     model.eval()
    
#     transform = T.Compose([T.ToTensor()])
#     image_tensor = transform(image)
#     with torch.no_grad():
#         predictions = model([image_tensor])
    
#     garnishing_label_id = 52  # You need to replace this with the actual label ID for garnishing
    
#     garnishing_count = sum(1 for pred in predictions[0]['labels'] if pred == garnishing_label_id)
    
#     return garnishing_count

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
    hue = np.mean(image_hsv[:, :, 0])
    saturation = np.mean(image_hsv[:, :, 1])
    brightness = np.mean(image_hsv[:, :, 2])
    
    # Here I'm just assuming some dummy dominant color extraction, you might want to replace this 
    # with a proper dominant color extraction method.
    dominant_colors = KMeans(n_clusters=5).fit(pixels).cluster_centers_[:3]
    
    return mean_rgb, dominant_colors, hue, saturation, brightness

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
#CALLING FUNCTIONS
# List to store the results as dictionaries
results_list = []

# Process each image
for idx, row in sampled_images_df.iterrows():
    shortcode = row['shortcode']
    caption = row['caption']
    # image_url = row['display_url']
    # Load the saved image
    image_path = os.path.join("/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/code/saved_images", f"{shortcode}.jpg")
    image = Image.open(image_path)
    # image = download_image(image_url)
    
    if image:
        print(f"Processing image {shortcode}")
        sharpness = calculate_sharpness(image)
        sharpn = calculate_sharpn(image)
        exposure = calculate_exposure(image)
        brilliance = calculate_brilliance(image)
        colorfulness = calculate_colorfulness(image)
        highlights = calculate_highlights(image)
        vibrancy = calculate_vibrancy(image)
        warmth = calculate_warmth(image)
        tint = calculate_tint(image)
        definition = calculate_definition(image)
        noise_reduction = calculate_noise_reduction(image)
        vignette = calculate_vignette(image)
        tone = calculate_tone(image)
        depth = estimate_depth(image)
        shadows = calculate_shadows(image)
        contrast = calculate_contrast(image)
        black_point = calculate_black_point(image)
        #Convert PIL image to OpenCV format for composition evaluation
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        symmetry_score = evaluate_symmetry(image_cv)
        horizontal, vertical, diagonal = evaluate_lines(image_cv)
        triangle_count = evaluate_triangles(image_cv)
        center_score = evaluate_center_composition(image_cv)
        
        # Extract RGB values
        mean_rgb, dominant_colors, hue, saturation, brightness = extract_rgb_values(image)
        
        # Detect the language of the caption
        # caption_lang = detect_language(caption)

        # Append results as dictionary to results_list
        results_list.append({
            'shortcode': shortcode,
            'sharpness': sharpness,
            'sharpn': sharpn,
            'exposure': exposure,
            'brilliance': brilliance,
            'colorfulness': colorfulness,
            'highlights': highlights,
            'vibrancy': vibrancy,
            'warmth': warmth,
            'tint': tint,
            'definition': definition,
            'noise_reduction': noise_reduction,
            'vignette': vignette,
            'tone': tone,
            'depth': depth, 
            'shadows': shadows,
            'contrast': contrast,
            'black_point': black_point,
            'mean_rgb': mean_rgb,
            'dominant_colors': dominant_colors,
            'hue': hue,
            'saturation': saturation,
            'brightness': brightness,
            'symmetry_score': symmetry_score,
            'lines_horizontal': horizontal,
            'lines_vertical': vertical,
            'lines_diagonal': diagonal,
            'triangle_count': triangle_count,
            'center_score': center_score
            # ,'caption_lang': caption_lang
        })
    else:
        print(f"Skipping image {shortcode} due to download error.")

#%%
# Convert results_list to DataFrame
results_df = pd.DataFrame(results_list)

# Merge the results with the original dataset
final_df = pd.merge(sampled_images_df, results_df, on='shortcode')

# %%
final_df.head()

#%%
#separate the bracket cols into individual cols
def split_rgb_list(df, col_name):
    # Split the RGB values into separate columns
    df[[f'{col_name}_r', f'{col_name}_g', f'{col_name}_b']] = pd.DataFrame(df[col_name].tolist(), index=df.index)
    return df

# Apply function to split mean_rgb column
final_df = split_rgb_list(final_df, 'mean_rgb')



#%%

# %%
#--------------------------------------------------------------------------------------------------------------------------------
#%%
#preprocess the dominant_colors column to get color names
def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)

def get_color_names_from_rgb_list(rgb_list):
    color_names = []
    for rgb in rgb_list:
        if isinstance(rgb, np.ndarray) and rgb.size == 3:
            rgb = tuple(rgb.astype(int))  # Convert to tuple and ensure integer values
        elif isinstance(rgb, list) or isinstance(rgb, tuple):
            rgb = tuple(int(val) for val in rgb)  # Convert list/tuple values to integers
        else:
            continue  # Skip if the RGB value is not valid
        color_names.append(get_color_name(rgb))
    return color_names


#%%
final_df['color_names'] = final_df['dominant_colors'].apply(get_color_names_from_rgb_list)

#%%
new_df = final_df.copy()

#%%
final_df.shape

#%%
final_df.head(10)

#%%
#--------------------------------------------------------------------------------------------------------------------------------
# %%

# Convert the 'tone' column to binary
final_df['tone'] = final_df['tone'].map({'Cool': 1, 'Warm': 0})
final_df.head()


#%%
#ENGAGEMENT METRIC

# # Convert timestamp to datetime
# final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])

# #find the time since post
# scrape_time = pd.to_datetime('2024-08-10 19:13:00')

# # Calculate time since post in hours
# final_df['time_since_post'] = ((scrape_time - final_df['timestamp']).dt.total_seconds() / 3600).round(2)

# # Define the growth rate constant
# # k = 2

# # f(time since post)- exponential growth function
# # df['exp_growth'] = (np.exp(k * df['time_since_post']) - 1).round(2)

# #%%
# #creating engagement metric
# # final_df['eng_met'] = ((final_df['like_count'] + (2 * final_df['comment_count'])) / ((final_df['followers'])
# # #+df['exp_growth']
# # )).round(2)

# #%%
# # Drop cols
# # final_df.drop(columns=['time_since_post', 'exp_growth'], inplace=True)
# final_df.drop(columns=['time_since_post'], inplace=True)
# final_df.head()

#%%
#%%
# Combine line columns
final_df['lines_count'] = final_df['lines_horizontal'] + final_df['lines_vertical'] + final_df['lines_diagonal']
# Combine RGB columns
final_df['mean_rgb'] = (final_df['mean_rgb_r'] + final_df['mean_rgb_g'] + final_df['mean_rgb_b']/3)
# Drop the original columns
final_df.drop(columns=['lines_horizontal', 'lines_vertical', 'lines_diagonal','triangle_count', 'mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b'], inplace=True)

final_df.head()

#%%
# #DATA SAMPLING - bootstrap sampling
# # Set the target number of rows
# target_rows = 600

# # Perform bootstrap sampling
# final_df = final_df.sample(n=target_rows, replace=True, random_state=1)

# # Save the new dataset to a CSV file
# # bootstrap_sampled_data.to_csv('expanded_model_data.csv', index=False)

# # Verify the shape of the new dataset
# print("Shape of augmented dataset:", final_df.shape)

# %%
#saving as csv
final_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/eng_met.csv', index=False)

#%%