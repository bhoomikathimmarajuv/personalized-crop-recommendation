import pandas as pd

# Load one-hot encoded data
data = pd.read_csv(r"C:\Users\Bhoomika T\OneDrive\Desktop\Randomforest\Crop_recommendation_onehot.csv")

# Convert one-hot encoded labels to single column
data['label'] = data.iloc[:, 7:].idxmax(axis=1)

# Keep only feature + label columns
data = data.iloc[:, :7].join(data['label'])

# Save clean version
data.to_csv(r"C:\Users\Bhoomika T\OneDrive\Desktop\Randomforest\crop_clean.csv", index=False)

print("✅ Clean dataset saved successfully!")
