from load_data import load_dataset

X, y, le = load_dataset()

print("Images loaded:", X.shape)
print("Labels loaded:", y.shape)
print("Class names:", le.classes_) 
