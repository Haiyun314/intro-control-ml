import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(0)


# Define the neural network for the heat equation (for temperature prediction)
class HeatEquationNN(nn.Module):
    def __init__(self):
        super(HeatEquationNN, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # Input: (x, y, t)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Output: temperature (u)

    def forward(self, x, y, t):
        input_tensor = torch.cat([x, y, t], dim=1)
        hidden = torch.relu(self.fc1(input_tensor))
        hidden = torch.relu(self.fc2(hidden))
        u = self.fc3(hidden)
        return u


# Loss function for the heat equation
def heat_equation_loss(model, x, y, t, alpha):
    u = model(x, y, t)  # Predict temperature (u)

    # Compute gradients
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    # Heat equation residual
    heat_residual = u_t - alpha * (u_xx + u_yy)

    # Loss = squared residual
    return torch.mean(heat_residual ** 2)


# Training loop
def train_heat_equation_model(model, optimizer, alpha, num_epochs=200):
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Sample random spatial and temporal points with requires_grad=True
        x = torch.range(0, 1, step=0.02, requires_grad=True) * 2 - 1  # x in [-1, 1]
        y = torch.range(0, 1, step=0.02, requires_grad=True) * 2 - 1  # y in [-1, 1]
        t = torch.range(0, 1, step=0.02, requires_grad=True)  # t in [0, 1]

        # Compute loss
        loss = heat_equation_loss(model, x, y, t, alpha)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
