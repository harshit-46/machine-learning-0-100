#To create a sample dummy file to test the DataLoader class

import pandas as pd
import numpy as np

# Create a small dummy dataset
df = pd.DataFrame({
    "id"        : range(1, 101),
    "age"       : np.random.randint(18, 65, 100),
    "salary"    : np.random.uniform(30000, 120000, 100).round(2),
    "city"      : np.random.choice(["Mumbai", "Delhi", "Bangalore", "Chennai"], 100),
    "score"     : np.random.randn(100).round(4),
})

# Introduce some nulls so validate() has something to find
df.loc[5:10, "age"] = np.nan
df.loc[20:22, "salary"] = np.nan

# Introduce a duplicate row
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Save it
df.to_csv("data/test_data.csv", index=False)
print("Dummy dataset created at data/test_data.csv")