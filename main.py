from timeline.Day1.data_loader import DataLoader

loader = DataLoader("test_model", "data/test_data.csv")
loader.load()
loader.validate()
loader.summary()