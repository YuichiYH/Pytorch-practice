import numpy as np

# Predict a Linear Function without bias only weight

# f = w * X
# f = 2 * X
# Input
X = np.array ([1,2,3,4], dtype=np.float32)

# Predicted Output
Y = np.array ([2,4,6,8], dtype=np.float32)

# Weight
w = 0.0

# forwardpass
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# d3/dw = 1/N 2x (w*x - y)

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = 0.01

n_iters = 10

for epoch in range(n_iters):
    #prediction = foward pass
    y_pred = forward(X)

    #loss
    l = loss(Y,y_pred)

    #gradient
    dw = gradient(X, Y, y_pred)

    #update
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
