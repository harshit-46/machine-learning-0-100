import numpy as np

# ---------------------
# Dataset
# ---------------------
X = np.array([[1], [2], [3], [4]])
y = np.array([[3], [5], [7], [9]])

# ---------------------
# Initialize parameters
# ---------------------
W = np.random.randn(1, 1)
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# ---------------------
# Training loop
# ---------------------
for i in range(epochs):

    # 1. Prediction
    y_pred = X @ W + b

    # 2. Loss
    loss = (1/n) * np.sum((y - y_pred)**2)

    # 3. Gradients
    dW = (-2/n) * X.T @ (y - y_pred)
    db = (-2/n) * np.sum(y - y_pred)

    # 4. Update
    W = W - learning_rate * dW
    b = b - learning_rate * db

    if(i % 100 == 0):
    # 5. Print everything
        print(f"y predicted : {y_pred}")
        print("Loss:", loss)
        print(f"dw is : {dW}")
        print(f"db is : {db}")
        print("Epoch:", i)
        print("Weight:", W)
        print("Bias:", b)
        print("-" * 40)

# ---------------------
# Final parameters
# ---------------------
print("Final Weight:", W)
print("Final Bias:", b)