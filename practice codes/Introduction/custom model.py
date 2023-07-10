import torch
import torch.nn as nn

# Predict a Linear Function without bias only weight

# f = w * X
# f = 2 * X
# Input
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)

# Predicted Output
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

# Weight
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# requires input and output size
#model = nn.Linear(input_size,output_size)

#Custom model
class LinearRegresssion(nn.Module):
    def __init__(self, input_dimesions, output_dimensions):
        super(LinearRegresssion, self).__init__()
        # Define layers
        self.lin = nn.Linear(input_dimesions, output_dimensions)

    def forward(self, x):
        return self.lin(x)
    
model = LinearRegresssion(input_size,output_size)

# forwardpass 
# def forward(x):
#     return w * x

# loss = MSE
# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

# MSE = Mean Square Error
loss = nn.MSELoss()

# gradient
# gradient
# w*x = predicted output
# y = correct output
# N = number of iterations, because its the mean(average)

# MSE = 1/N * (w*x - y)**2
# d3/dw = 1/N 2x (w*x - y)

# def gradient(x, y, y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

learning_rate = 0.1
n_iters = 200

# Stochastic gradient descent
# It needs parameters, the weights
# and the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = foward pass
    y_pred = model(X)

    #loss
    l = loss(Y,y_pred)

    #gradient
    l.backward()

    #update
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # Zero gradient
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
