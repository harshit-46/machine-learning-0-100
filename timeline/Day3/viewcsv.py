import polars as pl

df = pl.read_csv("../../data/yellow_tripdata_2016-03.csv")

print(df.head())