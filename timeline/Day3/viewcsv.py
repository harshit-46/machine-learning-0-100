import polars as pl

pl.Config.set_tbl_cols(20)

df = pl.read_csv("../../data/yellow_tripdata_2016-03.csv")

print(df.head())