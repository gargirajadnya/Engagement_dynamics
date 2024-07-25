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
directory_path = '/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data'

def sort_key(raw_response):
    return int(raw_response.split('-')[1].split('.')[0])

# Get a list of all CSV files in the directory and sort them using the custom key
csv_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.csv')], key=sort_key)

# Read and concatenate all CSV files
sampled_images_df = pd.concat([pd.read_csv(os.path.join(directory_path, file)) for file in csv_files], ignore_index=True)


#%%

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


#%%
# Function to detect the language
def detect_language(text):
    try:
        lang, _ = langid.classify(text)
        return lang
    except Exception:
        return 'na'
    
#%%
sampled_images_df = prep_colnames(sampled_images_df)
sampled_images_df.head()

#%%
# Apply the function to the 'text' column and create a new column with the extracted hashtags
sampled_images_df['hashtags'] = sampled_images_df['node_edge_media_to_caption_edges_0_node_text'].apply(extract_hashtags)
sampled_images_df['caption'] = sampled_images_df['node_edge_media_to_caption_edges_0_node_text'].apply(clean_text)

sampled_images_df['hashtags'] = sampled_images_df['hashtags'].replace('', 'NA')
sampled_images_df['caption'] = sampled_images_df['caption'].replace('', 'NA')

sampled_images_df['cap_lang'] = sampled_images_df['caption'].apply(detect_language)

# Display the DataFrame
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
    'node_taken_at_timestamp':'timestamp'
}
sampled_images_df.rename(columns=rename_dict, inplace=True)


#%%
# select required columns
selected_columns = ['shortcode', 'timestamp', 'like_count', 'comment_count', 'dim_h', 'dim_w', 'hashtags', 'caption', 'cap_lang', 'display_url']

# new DataFrame with the selected columns
new_df = sampled_images_df[selected_columns]
new_df.shape

#%%

#drop duplicates
final_df = new_df.drop_duplicates(subset='shortcode')
# Convert timestamp to datetime
final_df.loc[:, 'timestamp'] = final_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

final_df.shape

#%%
final_df.head(10)

# %%
final_df.to_csv('clean_data.csv', index=False)

# %%


