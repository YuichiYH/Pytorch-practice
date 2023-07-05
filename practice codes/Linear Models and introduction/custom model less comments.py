import torch
import torch.nn as nn

# Predict a Linear Function without bias only weight

# f = w * X
# f = 2 * X
# Input
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)

# Predicted Output
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

# Test prediction
X_test = torch.tensor([5],dtype=torch.float32)

# number of samples and features/inputs
n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

#Predetermined Linear Model
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

# MSE = Mean Square Error
loss = nn.MSELoss()

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
    optimizer.step()

    # Zero gradient
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
