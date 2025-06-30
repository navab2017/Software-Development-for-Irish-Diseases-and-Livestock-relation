from google.colab import files
uploaded = files.upload()
import pandas as pd

# Replace with actual filename from upload if needed
livestock_df = pd.read_csv("livestock_ireland.csv")
disease_df = pd.read_csv("disease_ireland.csv")

print(livestock_df.head())
print(disease_df.head())
# Check for missing values
print("Missing values per column:\n", livestock_df.isnull().sum())

# Count total rows
print("Total rows:", len(livestock_df))

# Check for unique animals
print("Unique animals:", livestock_df['Type of Animal'].unique())

# Group by year and animal
grouped = livestock_df.groupby(['year and Month', 'Type of Animal']).size().reset_index(name='counts')

# Check for unexpected duplicates or structural issues
duplicates = grouped[grouped['counts'] > 1]
print("Duplicate rows per year/month and animal:", duplicates)

# Find extra or invalid months or years
print(livestock_df['year and Month'].value_counts().head(10))  # Check for formatting issues
# Check number of rows and columns
print("Livestock shape:", livestock_df.shape)
print("Disease shape:", disease_df.shape)
# Check for missing years
print("Unique years in disease dataset:", disease_df["Year"].unique())

# Sort and check
print(disease_df["Year"].sort_values().to_list())
print(livestock_df.columns.tolist())
# Fix inconsistent column names
livestock_df.columns = (
    livestock_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[()/]", "", regex=True)
)

# Optional: Rename for clarity
livestock_df.rename(columns={
    "year_and_month": "date",
    "type_of_animal": "animal",
    "value_headnumber": "count_head",
    "valueweight": "weight_tonnes"
}, inplace=True)

# Check results
print(livestock_df.columns.tolist())
print(livestock_df.head())
# Split 'date' column into 'year' and 'month'
livestock_df[['year', 'month']] = livestock_df['date'].str.split(' ', expand=True)
livestock_df['year'] = livestock_df['year'].astype(int)

# Create a proper datetime column (optional but useful)
livestock_df['date_parsed'] = pd.to_datetime(livestock_df['date'], format='%Y %B', errors='coerce')

# Preview results
print(livestock_df[['date', 'year', 'month', 'date_parsed']].head())
# Drop rows that have neither head count nor weight
livestock_df.dropna(subset=['count_head', 'weight_tonnes'], how='all', inplace=True)

# Check shape
print("Cleaned livestock shape:", livestock_df.shape)
print(livestock_df.columns.tolist())
# Group by year and animal type, summing head count and weight
livestock_yearly = livestock_df.groupby(['year', 'animal'], as_index=False)[['count_head', 'weight_tonnes']].sum()

# Preview the result
print(livestock_yearly.head(10))
import seaborn as sns
import matplotlib.pyplot as plt

# Use seaborn's built-in theme directly
sns.set_theme(style="whitegrid")

# Plot
plt.figure(figsize=(16, 8))
for animal in livestock_yearly['animal'].unique():
    animal_data = livestock_yearly[livestock_yearly['animal'] == animal]
    plt.plot(animal_data['year'], animal_data['count_head'], label=animal)

plt.title("Livestock Headcount in Ireland Over Time (1975–2025)", fontsize=16)
plt.xlabel("Year")
plt.ylabel("Headcount (in thousands)")
plt.legend(title="Animal Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Drop rows with missing 'Year' and reassign properly
disease_df = disease_df.dropna(subset=['Year']).copy()

# Now safely convert 'Year' to int using .loc
disease_df.loc[:, 'Year'] = disease_df['Year'].astype(int)
# Confirm there are no missing values in 'Year' and that its type is now int
print(disease_df['Year'].isna().sum())  # should print 0
print(disease_df['Year'].dtype)         # should print int64
print(disease_df.head())                # preview cleaned data
import matplotlib.pyplot as plt
import seaborn as sns

# Example: plot total headcount vs. male cancer rate
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='count_head', y='cancers, males', hue='animal')
plt.title('Livestock Headcount vs. Male Cancer Rate')
plt.xlabel('Headcount')
plt.ylabel('Cancer Rate (Males)')
plt.grid(True)
plt.legend(title='Animal Type')
plt.show()
# Trend over years
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_df, x='year', y='count_head', hue='animal')
plt.title('Livestock Headcount Over Time')
plt.xlabel('Year')
plt.ylabel('Headcount')
plt.grid(True)
plt.show()
correlation = merged_df[['count_head', 'cancers, males', 'cancers, females']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm')
from sklearn.linear_model import LinearRegression
import numpy as np

X = merged_df[['count_head']].values
y = merged_df['cancers, males'].values

model = LinearRegression()
model.fit(X, y)

print("R^2 score:", model.score(X, y))
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Prepare features (you can add more like 'weight_tonnes', 'animal type' averages per year, etc.)
features = livestock_df.groupby('year')[['count_head']].sum()

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
features['cluster'] = kmeans.fit_predict(features)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(features.index, features['count_head'], c=features['cluster'], cmap='viridis')
plt.xlabel('Year')
plt.ylabel('Total Headcount')
plt.title('KMeans Clustering of Livestock Headcount by Year')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
