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


#%%
# Load your DataFrame
sampled_images_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/raw_response_1.csv')
sampled_images_df.head()

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
    return re.findall(r'#\w+', text)

#%%
# Function to extract clean text
def clean_text(text):
    # Remove hashtags and the words following them
    text = re.sub(r'#\w+', '', text)
    # Remove special characters, including backslashes and newline characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Remove multiple spaces and strip extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


#%%
sampled_images_df = prep_colnames(sampled_images_df)
sampled_images_df.head()

#%%
# Apply the function to the 'text' column and create a new column with the extracted hashtags
sampled_images_df['hashtags'] = sampled_images_df['node_edge_media_to_caption_edges_0_node_text'].apply(extract_hashtags)
sampled_images_df['cleaned_text'] = sampled_images_df['node_edge_media_to_caption_edges_0_node_text'].apply(clean_text)

# Display the DataFrame
sampled_images_df.head()

# %%
sampled_images_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/clean_data.csv', index=False)

# %%
