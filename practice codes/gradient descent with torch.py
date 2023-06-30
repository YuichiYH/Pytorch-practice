import torch

# Predict a Linear Function without bias only weight

# f = w * X
# f = 2 * X
# Input
X = torch.tensor([1,2,3,4], dtype=torch.float32)

# Predicted Output
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

# Weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# forwardpass 
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# d3/dw = 1/N 2x (w*x - y)
# def gradient(x, y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

learning_rate = torch.tensor(0.01, dtype=torch.float32)
n_iters = 100

for epoch in range(n_iters):
    #prediction = foward pass
    y_pred = forward(X)

    #loss
    l = loss(Y,y_pred)

    #gradient
    l.backward()

    #update
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Zero gradient
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
