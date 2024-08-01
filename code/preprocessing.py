#%%

#%%
#basic
import pandas as pd
import numpy as np
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

#%%
directory_path = '/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/raw_data'

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
sampled_images_df.shape

#%%
# Remove rows where followers == 999
sampled_images_df = sampled_images_df[sampled_images_df['followers'] != 999]
sampled_images_df.shape

#%%
# select required columns
selected_columns = ['shortcode', 'timestamp', 'like_count', 'comment_count', 'followers', 'dim_h', 'dim_w', 'hashtags', 'caption', 'display_url']

# new DataFrame with the selected columns
new_df = sampled_images_df[selected_columns]
new_df.shape

#%%
new_df.head(10)

# %%
new_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/clean_data.csv', index=False)

# %%


