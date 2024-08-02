#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Load your DataFrame
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/check.csv')
food_df.head()

#%%
# Check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)


# %%
# Select specific columns for correlation, if needed
col_int = ['sharpness',
       'colorfulness', 'number_of_colors', 'garnishing', 'depth',
       'clarity', 'hue', 'saturation',
       'brightness', 'rule_of_thirds_x', 'rule_of_thirds_y', 'symmetry_score',
       'lines_horizontal', 'lines_vertical', 'lines_diagonal', 'pattern_score',
       'triangle_count', 'center_score', 'mean_rgb_r',
       'mean_rgb_g', 'mean_rgb_b'] 
correlation_matrix = food_df[col_int].corr()

# Create a colormap from the 'mako' color palette
mako_cmap = sns.color_palette("Blues", as_cmap=True)

# Plotting the heatmap with the 'mako' colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap with Mako Colormap')
plt.show()

# %%
