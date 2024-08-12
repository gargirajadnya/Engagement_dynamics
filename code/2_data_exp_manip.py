#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy.stats import ttest_ind

#%%
# Load your DataFrame
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/eng_met.csv')
food_df.head()

#%%
# Check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)

#%%
# Replace NaN values in eng_met with 0
food_df['eng_met'] = np.nan_to_num(food_df['eng_met'], nan=0)

#%%
#let's drop unnecessary columns
cols_drop = ['node_comments_disabled', 'node_typename', 'node_id', 'raw_caption',
       'shortcode', 'timestamp','node_edge_media_preview_like_count',
       'node_owner_id', 'node_thumbnail_src', 'node_thumbnail_resources_0_src',
       'node_thumbnail_resources_0_config_width',
       'node_thumbnail_resources_0_config_height',
       'node_thumbnail_resources_1_src',
       'node_thumbnail_resources_1_config_width',
       'node_thumbnail_resources_1_config_height',
       'node_thumbnail_resources_2_src',
       'node_thumbnail_resources_2_config_width',
       'node_thumbnail_resources_2_config_height',
       'node_thumbnail_resources_3_src',
       'node_thumbnail_resources_3_config_width',
       'node_thumbnail_resources_3_config_height',
       'node_thumbnail_resources_4_src',
       'node_thumbnail_resources_4_config_width',
       'node_thumbnail_resources_4_config_height', 'node_is_video',
       'node_accessibility_caption', 'node_product_type',
       'node_video_view_count', 'hashtags', 'caption','sharpn']

food_df = food_df.drop(columns = cols_drop)

# %%
# Select specific columns for correlation, if needed
num_f = food_df.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", num_f)

col_int = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met'] 


correlation_matrix = food_df[col_int].corr()

# Create a colormap from the 'mako' color palette
mako_cmap = sns.color_palette("Blues", as_cmap=True)

# Plotting the heatmap with the 'mako' colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap with Mako Colormap')
plt.show()


#%%

#let's check for multicollinearity and significant variables
# Calculate VIF scores for each feature
X = food_df[col_int].dropna()

# Add a constant column for statsmodels
X = sm.add_constant(X)

# Create a DataFrame to store VIF scores
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Drop the constant column VIF score
vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

high_vif_features = vif_data[vif_data['VIF'] > 10]
print(high_vif_features)

#%%
#feature engineering
#color names- one hot encoding
# Select the specified columns
selected_df = food_df.copy()

# One-hot encode 'caption_lang' using 1 and 0
# caption_lang_encoded = pd.get_dummies(selected_df['caption_lang'], prefix='lang').astype(int)

# Convert the list of colors in 'color_names' to individual columns for one-hot encoding
color_names_exploded = selected_df['color_names'].apply(lambda x: x.strip("[]").replace("'", "").split(', ') if pd.notna(x) else [])

# Get only the top 3 dominant colors
top_colors_exploded = color_names_exploded.apply(lambda x: x[:3])

# Collect unique colors from the top 3
unique_colors = set(color for sublist in top_colors_exploded for color in sublist)

# Convert the set of unique colors to a sorted list for DataFrame columns
unique_colors_list = sorted(list(unique_colors))

# Initialize DataFrame for color one-hot encoding
color_one_hot = pd.DataFrame(0, index=selected_df.index, columns=unique_colors_list)

# Populate the color one-hot encoding DataFrame
for i, colors in enumerate(top_colors_exploded):
    for color in colors:
        if color in color_one_hot.columns:
            color_one_hot.loc[i, color] = 1

# Concatenate the one-hot encoded columns with the selected DataFrame

food_df = pd.concat([selected_df
# , caption_lang_encoded
, color_one_hot], axis=1).drop([ 'color_names'
                                # ,'caption_lang'
                                , 'dominant_colors'], axis=1)

# Display the first few rows of the resulting DataFrame
food_df.head()

#%%
# -------------------------------------------------------------------------
food_df.to_csv("/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv")

model_df = food_df.copy()

# %%
outlier_df = model_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met']]
outlier_df.shape

#%%

# Initialize a DataFrame to keep track of rows to remove
rows_to_remove = pd.Series([False] * len(model_df))

# Dictionary to keep track of the number of outliers per column
outlier_counts = {}

# Iterate through each numeric column to detect outliers
numeric_columns = outlier_df.select_dtypes(include=['float64', 'int64']).columns

for column_name in numeric_columns:
    # Calculate Q1, Q3, and IQR for outlier detection
    Q1 = outlier_df[column_name].quantile(0.25)
    Q3 = outlier_df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    is_outlier = (outlier_df[column_name] < (Q1 - 1.5 * IQR)) | (outlier_df[column_name] > (Q3 + 1.5 * IQR))
    
    # Count outliers
    num_outliers = is_outlier.sum()
    outlier_counts[column_name] = num_outliers

    # Update rows_to_remove series
    rows_to_remove = rows_to_remove | is_outlier

    # Create a box plot for visualization
    plt.figure(figsize=(10, 6))
    plt.boxplot(outlier_df[column_name], vert=False)
    plt.title(f'Box plot for {column_name}')
    plt.xlabel(column_name)
    plt.show()

# Remove the rows with any outliers
model_df = model_df[~rows_to_remove]

# Display the number of outliers per column
print("Number of outliers per column:")
for column_name, count in outlier_counts.items():
    print(f"{column_name}: {count}")

# Display the shape of the resulting DataFrame
print(f"Shape of DataFrame after removing outliers: {model_df.shape}")

#%%
#DATA SAMPLING - bootstrap sampling
# Set the target number of rows
target_rows = 1000

# Perform bootstrap sampling
model_df = model_df.sample(n=target_rows, replace=True, random_state=1)

# Save the new dataset to a CSV file
# bootstrap_sampled_data.to_csv('expanded_model_data.csv', index=False)

# Verify the shape of the new dataset
print("Shape of augmented dataset:", model_df.shape)

#%%
#save data in csv
model_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/bootstrapped_df.csv', index=False)

