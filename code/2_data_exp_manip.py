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

#standardization
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#splitting data
from sklearn.model_selection import train_test_split

#%%
# Load your DataFrame
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/eng_met.csv')
food_df.head()

#%%
# Check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)


#%%
#new variables
#Interaction term between brightness and saturation
food_df['brns_satrn_interaction'] = food_df['brightness'] * food_df['saturation']

#Interaction between 'depth' and 'clarity'
food_df['depth_clarity_interaction'] = food_df['depth'] * food_df['clarity']

#Log transform for 'symmetry_score'
food_df['log_symmetry_score'] = np.log1p(food_df['symmetry_score'])

# %%
# Select specific columns for correlation, if needed
# col_int = ['sharpness', 'colorfulness', 'depth', 'clarity', 'hue', 'saturation', 'brightness', 'rule_of_thirds_x', 'rule_of_thirds_y', 'symmetry_score', 'tone','lines_horizontal', 'lines_vertical', 'lines_diagonal', 'triangle_count', 'center_score', 'mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b', 'brns_satrn_interaction', 'depth_clarity_interaction', 'log_symmetry_score'] 

col_int =  ['colorfulness', 'depth', 'clarity', 'hue', 'saturation', 'brightness', 'rule_of_thirds_x', 'rule_of_thirds_y', 'symmetry_score',  'lines_vertical', 'center_score', 'tone', 'brns_satrn_interaction', 'depth_clarity_interaction', 'log_symmetry_score'] 

correlation_matrix = food_df[col_int].corr()

# Create a colormap from the 'mako' color palette
mako_cmap = sns.color_palette("Blues", as_cmap=True)

# Plotting the heatmap with the 'mako' colormap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap with Mako Colormap')
plt.show()

# %%
#columns to select after checking correlation
sel_num_col = ['sharpness', 'colorfulness', 'depth', 'clarity', 'hue', 'saturation', 'brightness', 'rule_of_thirds_x', 'rule_of_thirds_y', 'symmetry_score', 'tone', 'lines_horizontal', 'lines_vertical', 'lines_diagonal', 'triangle_count', 'center_score', 'mean_rgb_r', 'mean_rgb_g', 'mean_rgb_b'] 


#%%
#let's check for multicollinearity and significant variables
# Calculate VIF scores for each feature
X = food_df[sel_num_col].dropna()

# Add a constant column for statsmodels
X = sm.add_constant(X)

# Create a DataFrame to store VIF scores
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Drop the constant column VIF score
vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

print(vif_data)

# # Plot VIF scores
# plt.figure(figsize=(10, 8))
# sns.barplot(x='VIF', y='Feature', data=vif_data, palette='Blues_d')
# plt.title('Variance Inflation Factor (VIF) Scores')
# plt.xlabel('VIF Score')
# plt.ylabel('Features')
# plt.show()

# T-Test for each variable
if 'group' not in food_df.columns:
    food_df['group'] = np.random.choice(['A', 'B'], size=len(food_df))  # Random binary groups for demonstration

# Perform t-test for each variable
# Initialize a list to store the results
results = []

# Perform t-tests for each column
for col in col_int:
    group1 = food_df[food_df['group'] == 'A'][col].dropna()
    group2 = food_df[food_df['group'] == 'B'][col].dropna()
    
    if len(group1) > 1 and len(group2) > 1:  # Ensure there are enough samples in each group
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)  # Welch’s t-test
        results.append({
            'Column': col,
            'T-Statistic': t_stat,
            'P-Value': p_val
        })
    else:
        results.append({
            'Column': col,
            'T-Statistic': None,
            'P-Value': None
        })

# Convert the results list to a DataFrame
ttest_df = pd.DataFrame(results)
ttest_df

food_df.drop(columns=['group'], inplace=True)

#%%
#feature engineering
#color names- one hot encoding
# Select the specified columns
selected_df = food_df.copy()

# One-hot encode 'caption_lang' using 1 and 0
caption_lang_encoded = pd.get_dummies(selected_df['caption_lang'], prefix='lang').astype(int)

# Convert the list of colors in 'color_names' to individual columns for one-hot encoding
color_names_exploded = selected_df['color_names'].apply(lambda x: x.strip("[]").replace("'", "").split(', '))

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
result_df = pd.concat([selected_df, caption_lang_encoded, color_one_hot], axis=1).drop(['caption_lang', 'color_names', 'dominant_colors'], axis=1)

# Display the first few rows of the resulting DataFrame
result_df.head()

#%%
#save data in csv
result_df.to_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_data.csv', index=False)

# %%
