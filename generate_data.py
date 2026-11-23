import pandas as pd
import numpy as np
from crop_ranges import crop_ranges

data = []

# Generate 150 random samples per crop
for crop, vals in crop_ranges.items():
    for _ in range(150):
        N = np.random.uniform(*vals["N"])
        P = np.random.uniform(*vals["P"])
        K = np.random.uniform(*vals["K"])
        temperature = np.random.uniform(*vals["temperature"])
        humidity = np.random.uniform(*vals["humidity"])
        ph = np.random.uniform(*vals["ph"])
        rainfall = np.random.uniform(*vals["rainfall"])
        data.append([N, P, K, temperature, humidity, ph, rainfall, crop])

df = pd.DataFrame(data, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"])
df.to_csv("crop_range_data.csv", index=False)

print("✅ Synthetic dataset created successfully!")
print(df.head())
