import numpy as np
import sys
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import gamspy as gp

dfs = []

for letter in ["a", "b"]:
    for num in [1, 2, 3, 4]:
        name = f"{letter}_{num}.csv"
        dfs.append(pd.read_csv(name))

df = pd.concat(dfs)

input_cols = ["reactor", "fa", "fb", "fe"]
output_col = ["ok"]


x = df[input_cols].to_numpy()
y = df[output_col].to_numpy()


x_mean, x_std = x.mean(axis=0), x.std(axis=0)

x = (x - x_mean) / x_std
x_lb, x_ub = x.min(axis=0), x.max(axis=0)

x = torch.Tensor(x)
y = torch.Tensor(y)

all_data = TensorDataset(x, y)
train_dataset, validation_dataset = torch.utils.data.random_split(all_data, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=256)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10, 15)
        self.l4 = nn.Linear(15, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        relu = nn.ReLU()
        x = self.l1(x)
        x = relu(x)
        x = self.l2(x)
        x = relu(x)
        x = self.l3(x)
        x = relu(x)
        x = self.l4(x)
        x = self.sigmoid(x)
        return x


def train(model, train_loader, optimizer, epoch):
    loss_fn = nn.BCELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}"
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


def test(model, test_loader):
    loss_fn = nn.BCELoss(reduction="sum")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.round()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model = NeuralNetwork()

optimizer = optim.Adadelta(model.parameters(), lr=1)

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
for epoch in range(1, 25 + 1):
    train(model, train_loader, optimizer, epoch)
    test(model, validation_loader)
    scheduler.step()



m = gp.Container()

with torch.no_grad():
    relu = gp.math.relu_with_binary_var

    lin1 = gp.formulations.Linear(m, in_features=4, out_features=10)
    lin1.load_weights(model.l1.weight.numpy(), model.l1.bias.numpy())

    lin2 = gp.formulations.Linear(m, in_features=10, out_features=10)
    lin2.load_weights(model.l2.weight.numpy(), model.l2.bias.numpy())

    lin3 = gp.formulations.Linear(m, in_features=10, out_features=15)
    lin3.load_weights(model.l3.weight.numpy(), model.l3.bias.numpy())

    lin4 = gp.formulations.Linear(m, in_features=15, out_features=1)
    lin4.load_weights(model.l4.weight.numpy(), model.l4.bias.numpy())

a0 = gp.Variable(m, name="a0", domain=gp.math.dim([4]))  # input to neural network

a1 = gp.Variable(m, name="a1", domain=gp.math.dim([4])) # normalized

x_mean_par = gp.Parameter(
    m,
    name="x_mean_par",
    domain=gp.math.dim([4]),
    records=x_mean,
)

x_std_par = gp.Parameter(
    m,
    name="x_std_par",
    domain=gp.math.dim([4]),
    records=x_std,
)

normalize_input = gp.Equation(m, name="normalize_input", domain=a0.domain)
normalize_input[...] = a1 == (a0 - x_mean_par) / x_std_par

a1_lb = gp.Parameter(m, name="a1_lb", domain=a1.domain, records=x_lb)
a1_ub = gp.Parameter(m, name="a1_ub", domain=a1.domain, records=x_ub)
a1.lo[...] = a1_lb
a1.up[...] = a1_ub


z2, _ = lin1(a1)
a2, _ = relu(z2)

z3, _ = lin2(a2)
a3, _ = relu(z3)

z4, _ = lin3(a3)
a4, _ = relu(z4)

z5, _ = lin4(a4)

check_feasibility = gp.Equation(m, name="check_feasibility")
check_feasibility[...] = z5[0] >= 4.59511985013459 # 0.99 probability


reactors = gp.Set(m, name="r", records=["r1", "r2", "r3", "r4"])
active_reactor = gp.Variable(m, name="x_r", domain=[reactors], type="binary")


reactor_costs = gp.Parameter(
    m,
    name="r_cost",
    domain=[reactors],
    records=[("r1", 400), ("r2", 850), ("r3", 1200), ("r4", 1650)]
)

reactor_m3 = gp.Parameter(
    m,
    name="r_m3",
    domain=[reactors],
    records=[("r1", 5), ("r2", 20), ("r3", 35), ("r4", 50)]
)


separators = gp.Set(m, name="s", records=["s1", "s2", "s3", "s4"])
active_separator = gp.Variable(m, name="x_s", domain=[separators], type="binary")

separators_costs = gp.Parameter(
    m,
    name="s_cost",
    domain=[separators],
    records=[("s1", 300), ("s2", 720), ("s3", 980), ("s4", 1210)]
)

separators_min = gp.Parameter(
    m,
    name="s_min",
    domain=[separators],
    records=[("s1", 30), ("s2", 40), ("s3", 60), ("s4", 90)]
)

separators_max = gp.Parameter(
    m,
    name="s_max",
    domain=[separators],
    records=[("s1", 50), ("s2", 70), ("s3", 100), ("s4", 140)]
)

cost = gp.Variable(m, name="cost")

material_cost_per_mol = 20

set_cost = gp.Equation(m, name="set_cost")
set_cost[...] = cost == (
    gp.Sum(reactors, active_reactor[reactors] * reactor_costs[reactors]) +
    gp.Sum(separators, active_separator[separators] * separators_costs) +
    20 * a0[1]
)

pick_separator = gp.Equation(m, name="pick_separator")
pick_separator[...] = gp.Sum(separators, active_separator[separators]) == 1

pick_reactor = gp.Equation(m, name="pick_reactor")
pick_reactor[...] = gp.Sum(reactors, active_reactor[reactors]) == 1

set_reactor_input = gp.Equation(m, name="set_reactor_input")
set_reactor_input[...] = a0[0] == gp.Sum(reactors, reactor_m3[reactors] * active_reactor[reactors])

flow_separator_min = gp.Equation(m, name="flow_separator_min", domain=[separators])
flow_separator_min[...] = a0[1] + (1 - active_separator[separators]) * 200 >= separators_min[separators]

flow_separator_max = gp.Equation(m, name="flow_separator_max", domain=[separators])
flow_separator_max[...] = a0[1] <= separators_max[separators] + (1 - active_separator[separators]) * 200

a0.fx[2] = 55 # Fb demand
a0.fx[3] = 47 # Fe demand

opt_model = gp.Model(
    m,
    name="min_cost",
    equations=m.getEquations(),
    sense="min",
    objective=cost,
    problem="mip"
)
opt_model.solve()

print("Cost: ", cost.toDense())
print("Reactor capacity: ", a0.toDense()[0])
print("Fa0 capacity: ", a0.toDense()[1])
print("Fb demand: ", a0.toDense()[2])
print("Fe demand: ", a0.toDense()[3])

separator_index = np.where(active_separator.toDense())[0][0]
print("Active separator: ", separator_index + 1)

min_vals = [30, 40, 60, 90]
max_vals = [50, 70, 100, 140]

print(min_vals[separator_index], "<= (Fa0 =", a0.toDense()[1], ") <=", max_vals[separator_index])
