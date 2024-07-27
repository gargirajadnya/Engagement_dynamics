#%%

#%% 
#basic library load
import pandas as pd
import numpy as np

#%%
# Load DataFrame
final_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/final_data.csv')
final_df.head()

#%%
#load follower data
follower_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/followers.csv')

#%%
# Merging both dataframes on shortcode
df = pd.merge(final_df, follower_df, on='shortcode')
df.head()
# %%
