#%%
#delete the followers merge code after getting the data

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
# df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/check.csv', index=False)

#%%
# Convert columns to integers
df['like_count'] = df['like_count'].fillna(0).astype(int)
df['comment_count'] = df['comment_count'].fillna(0).astype(int)
df['followers'] = df['followers'].fillna(0).astype(int)

# %%
#creating engagement metric
# df['eng_met'] = (((df['like_count'] + df['comment_count']) / (df['followers']))*100).round(2)

# %%
df = pd.DataFrame(df)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# %%

# Constants for Bayesian Prior
prior_alpha = 2  # Prior alpha parameter for Beta distribution
prior_beta = 5   # Prior beta parameter for Beta distribution

current_time = pd.to_datetime('2024-07-25 19:13:00')

# Calculate time since post in hours
df['time_since_post'] = (current_time - df['timestamp']).dt.total_seconds() / 3600

#%%
# Calculate basic engagement score
df['basic_engagement_score'] = (df['like_count'] + 2 * df['comment_count']) / df['followers']

#%%

# Bayesian Update: Calculate posterior parameters
df['posterior_alpha'] = prior_alpha + df['like_count'] + 2 * df['comment_count']
df['posterior_beta'] = prior_beta + df['followers'] - (df['like_count'] + 2 * df['comment_count'])

# Calculate the mean of the posterior Beta distribution
df['posterior_mean'] = df['posterior_alpha'] / (df['posterior_alpha'] + df['posterior_beta'])

# Calculate time-weight adjustment factor
beta_adjustment = 2
gamma_adjustment = 0.1
df['time_weight_adjustment'] = 1 + (beta_adjustment / (1 + gamma_adjustment * df['time_since_post']))

# Calculate the adjusted engagement score
df['adjusted_engagement_score'] = df['posterior_mean'] * df['time_weight_adjustment']

# Drop intermediate columns if needed
# df = df.drop(columns=['time_since_post', 'posterior_alpha', 'posterior_beta', 'posterior_mean', 'time_weight_adjustment'])

df.head()
df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/check.csv', index=False)

#%%