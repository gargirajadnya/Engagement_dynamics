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

def sort_key(raw_response):
    return int(raw_response.split('-')[1].split('.')[0])

# Get a list of all CSV files in the directory and sort them using the custom key
csv_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.csv')], key=sort_key)

# Read and concatenate all CSV files
sampled_images_df = pd.concat([pd.read_csv(os.path.join(directory_path, file)) for file in csv_files], ignore_index=True)

#load follower data
follower_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/followers.csv')

#%%
#drop duplicates
sampled_images_df = sampled_images_df.drop_duplicates(subset='node/shortcode')
sampled_images_df.shape

#%%
# Merging both dataframes on shortcode
sampled_images_df = pd.merge(sampled_images_df, follower_df, on='node/shortcode')
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
    hashtags = re.findall(r'#\w+', text)
    return ', '.join(hashtags)

#%%
# Function to extract clean text
def clean_text(text):
    # Remove hashtags and the words following them
    text = re.sub(r'#\w+', '', text)
    
    # Remove all special symbols (non-alphanumeric and non-whitespace characters)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    # Remove multiple spaces and strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
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
sampled_images_df['followers'] = sampled_images_df['followers'].fillna(0).astype(int)

#%%
# Convert timestamp to datetime
sampled_images_df.loc[:, 'timestamp'] = sampled_images_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

#%%
sampled_images_df_copy = sampled_images_df.copy()
sampled_images_df.shape

#%%
# Remove rows where followers == 999
sampled_images_df = sampled_images_df[sampled_images_df['followers'] != 999]

#%%
#SAMPLING
# Filter the DataFrame to include only rows where the timestamp is earlier than '25/07/2024'
sampled_images_df = sampled_images_df[sampled_images_df['timestamp'] < '2024-07-25 00:00:00']

sampled_images_df.shape

#%%

# %%
#--------------------------------------------------------------------------------------------------------------------------------
#%%
#preprocess the dominant_colors column to get color names
def replace_array_string(s):
    return s.replace('array', 'np.array')

# Apply the function to the 'dominant_colors' column
sampled_images_df['dominant_colors'] = sampled_images_df['dominant_colors'].apply(replace_array_string)

# Use the updated parse function
def parse_array_string(array_string):
    array_string = array_string.replace('np.array(', '').replace(')', '')
    try:
        array_list = ast.literal_eval(array_string)
        return np.array(array_list)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string: {array_string} -> {e}")
        return np.array([])

sampled_images_df['dominant_colors'] = sampled_images_df['dominant_colors'].apply(parse_array_string)

#%%
#color names
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
        # Convert numpy array to tuple if necessary
        if isinstance(rgb, np.ndarray):
            rgb = tuple(rgb.astype(int))  # Convert to tuple and ensure integer values
        color_names.append(get_color_name(rgb))
    return color_names


#%%
sampled_images_df['color_names'] = sampled_images_df['dominant_colors'].apply(get_color_names_from_rgb_list)

#%%
new_df = sampled_images_df.copy()

#%%
# select required columns
selected_columns = ['shortcode', 'timestamp', 'like_count', 'comment_count', 'followers', 'dim_h', 'dim_w', 'hashtags', 'caption', 'display_url']

# new DataFrame with the selected columns
sampled_images_df = sampled_images_df[selected_columns]
sampled_images_df.shape

#%%
sampled_images_df.head(10)

# %%
# sampled_images_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/clean_data.csv', index=False)

# %%
#--------------------------------------------------------------------------------------------------------------------------------
#%%
#Image recognition and aesthetics
# Function to download an image from a URL
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None

# image = download_image(image_url)

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
#CALLING FUNCTIONS
# List to store the results as dictionaries
results_list = []

# Process each image
for idx, row in sampled_images_df.iterrows():
    shortcode = row['shortcode']
    caption = row['caption']
    image_url = row['display_url']
    
    image = download_image(image_url)
    if image:
        print(f"Processing image {shortcode}")
        sharpness = calculate_sharpness(image)
        colorfulness = calculate_colorfulness(image)
        number_of_colors = calculate_number_of_colors(image)
        tone = calculate_tone(image)
        # garnishing = detect_garnishing(image)
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
        
        # Detect the language of the caption
        caption_lang = detect_language(caption)

        # Append results as dictionary to results_list
        results_list.append({
            'shortcode': shortcode,
            'sharpness': sharpness,
            'colorfulness': colorfulness,
            'number_of_colors': number_of_colors,
            'tone': tone,
            # 'garnishing': garnishing,
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
            'center_score': center_score,
            'caption_lang': caption_lang
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
sampled_images_df = split_rgb_list(final_df, 'mean_rgb')



# %%
# sampled_images_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/im_aesth.csv', index=False)


#%%
#--------------------------------------------------------------------------------------------------------------------------------
# %%

# Convert the 'tone' column to binary
sampled_images_df['tone'] = sampled_images_df['tone'].map({'Cool': 1, 'Warm': 0})
sampled_images_df.head()

#ENGAGEMENT METRIC

# Convert timestamp to datetime
sampled_images_df['timestamp'] = pd.to_datetime(sampled_images_df['timestamp'])

#find the time since post
scrape_time = pd.to_datetime('2024-07-25 19:13:00')

# Calculate time since post in hours
sampled_images_df['time_since_post'] = ((scrape_time - sampled_images_df['timestamp']).dt.total_seconds() / 3600).round(2)

# Define the growth rate constant
# k = 2

# f(time since post)- exponential growth function
# df['exp_growth'] = (np.exp(k * df['time_since_post']) - 1).round(2)

#%%
#creating engagement metric
sampled_images_df['eng_met'] = ((sampled_images_df['like_count'] + (2 * sampled_images_df['comment_count'])) / ((sampled_images_df['followers'])
#+df['exp_growth']
)).round(2)

#%%
# Drop cols
# sampled_images_df.drop(columns=['time_since_post', 'exp_growth'], inplace=True)
sampled_images_df.drop(columns=['time_since_post'], inplace=True)
sampled_images_df.head()

#%%
#%%
# Combine line columns
sampled_images_df['lines_count'] = sampled_images_df['lines_horizontal'] + sampled_images_df['lines_vertical'] + sampled_images_df['lines_diagonal']
# Combine RGB columns
sampled_images_df['mean_rgb'] = (sampled_images_df['mean_rgb_r'] + sampled_images_df['mean_rgb_g'] + sampled_images_df['mean_rgb_b']/3)
# Drop the original columns
sampled_images_df.drop(columns=['lines_horizontal', 'lines_vertical', 'lines_diagonal','triangle_count', 'mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b'], inplace=True)

sampled_images_df.head()

#%%
#DATA SAMPLING - bootstrap sampling
# Set the target number of rows
target_rows = 600

# Perform bootstrap sampling
sampled_images_df = sampled_images_df.sample(n=target_rows, replace=True, random_state=1)

# Save the new dataset to a CSV file
# bootstrap_sampled_data.to_csv('expanded_model_data.csv', index=False)

# Verify the shape of the new dataset
print("Shape of augmented dataset:", sampled_images_df.shape)

# %%
#saving as csv
sampled_images_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/eng_met.csv', index=False)

#%%