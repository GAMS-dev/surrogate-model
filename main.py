import gamspy as gp
import pandas as pd

import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

import torch_sequential

doTraining = True
# doTraining = False

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

if doTraining:
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    all_data = TensorDataset(x, y)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        all_data, [0.8, 0.2]
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=256)

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

        print(
            "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    model = nn.Sequential(
        nn.Linear(4, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, 1),
        nn.Sigmoid()
    )

    optimizer = optim.Adadelta(model.parameters(), lr=1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 25 + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, validation_loader)
        scheduler.step()

    torch.save(model, "rs.pth")
    # sys.exit(0)
else:
    model = torch.load("rs.pth", weights_only=False)
    model.eval()


# return the label for the one active element of v(j)
def getActive(v: gp.Variable):
    label = "??label??"
    found = False
    for _, row in v.records.iterrows():
        rval = round(row.iloc[1], 3)
        if 0 == rval:
            continue
        assert 1 == rval, "getActive: nonzero binary values must be 1"
        assert (
            not found
        ), f"We already found that {label} is active: {row.iloc[0]} cannot be"
        label = row.iloc[0]
        found = True
    assert found, f"No active labels found in variable {v.name}"
    return (label, 1 + v.domain[0].toList().index(label))


m = gp.Container()

a0 = gp.Variable(m, name="a0", domain=gp.math.dim([4]))  # input to neural network
a1 = gp.Variable(m, name="a1", domain=gp.math.dim([4]))  # normalized

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


drop_sigmoid_model = nn.Sequential(*list(model.children())[:-1])

seq_formulation = torch_sequential.TorchSequential(m, drop_sigmoid_model)
z5, _ = seq_formulation(a1)

check_feasibility = gp.Equation(m, name="check_feasibility")
check_feasibility[...] = z5[0] >= 4.59511985013459  # 0.99 probability


reactors = gp.Set(m, name="r", records=["r1", "r2", "r3", "r4"])
active_reactor = gp.Variable(m, name="x_r", domain=[reactors], type="binary")


reactor_cost = gp.Parameter(
    m,
    name="r_cost",
    domain=[reactors],
    records=[("r1", 400), ("r2", 850), ("r3", 1200), ("r4", 1650)],
)

reactor_m3 = gp.Parameter(
    m,
    name="r_m3",
    domain=[reactors],
    records=[("r1", 5), ("r2", 20), ("r3", 35), ("r4", 50)],
)


separators = gp.Set(m, name="s", records=["s1", "s2", "s3", "s4"])
active_separator = gp.Variable(m, name="x_s", domain=[separators], type="binary")

separator_cost = gp.Parameter(
    m,
    name="s_cost",
    domain=[separators],
    records=[("s1", 300), ("s2", 720), ("s3", 980), ("s4", 1210)],
)

separator_min = gp.Parameter(
    m,
    name="s_min",
    domain=[separators],
    records=[("s1", 30), ("s2", 40), ("s3", 60), ("s4", 90)],
)

separator_max = gp.Parameter(
    m,
    name="s_max",
    domain=[separators],
    records=[("s1", 50), ("s2", 70), ("s3", 100), ("s4", 140)],
)

separator_bigM = gp.Parameter(m, domain=[separators])
separator_bigM[separators] = (
    separator_max.records.max().value - separator_max[separators]
)

cost = gp.Variable(m, name="cost")

material_cost_per_mol = 20

set_cost = gp.Equation(m, name="set_cost")
set_cost[...] = cost == (
    gp.Sum(reactors, active_reactor[reactors] * reactor_cost[reactors])
    + gp.Sum(separators, active_separator[separators] * separator_cost[separators])
    + 20 * a0[1]
)

pick_separator = gp.Equation(m, name="pick_separator")
pick_separator[...] = gp.Sum(separators, active_separator[separators]) == 1

pick_reactor = gp.Equation(m, name="pick_reactor")
pick_reactor[...] = gp.Sum(reactors, active_reactor[reactors]) == 1

set_reactor_input = gp.Equation(m, name="set_reactor_input")
set_reactor_input[...] = a0[0] == gp.Sum(
    reactors, reactor_m3[reactors] * active_reactor[reactors]
)

flow_separator_min = gp.Equation(m, name="flow_separator_min", domain=[separators])
flow_separator_min[...] = a0[1] >= active_separator * separator_min[...]


flow_separator_max = gp.Equation(m, name="flow_separator_max", domain=[separators])
flow_separator_max[...] = (
    a0[1] <= separator_max + (1 - active_separator) * separator_bigM
)

a0.fx[2] = 55  # Fb demand
a0.fx[3] = 47  # Fe demand

opt_model = gp.Model(
    m,
    name="min_cost",
    equations=m.getEquations(),
    sense="min",
    objective=cost,
    problem="mip",
)
opt_model.solve(output=sys.stdout)
# sys.exit(0)

print("Cost: ", cost.toDense())
print("Reactor capacity: ", a0.toDense()[0])
print("Fa0 capacity: ", a0.toDense()[1])
print("Fb demand: ", a0.toDense()[2])
print("Fe demand: ", a0.toDense()[3])

label, _ = getActive(active_separator)
print("Active separator: ", label)

tmp = gp.Parameter(m, domain=[])
tmp[...] = gp.Sum(
    separators,
    separator_min[separators] * gp.math.Round(active_separator.l[separators]),
)
minv = tmp.toValue()
tmp[...] = gp.Sum(
    separators,
    separator_max * gp.math.Round(active_separator.l),
)
maxv = tmp.toValue()

print(f"{minv} <= (Fa0={a0.toDense()[1]}) <= {maxv}")
print("")
# sys.exit(0)

# Define possible demand ranges
Fb_values = np.arange(10, 101, 10)
Fe_values = np.arange(10, 51, 10)
print("Now we solve for a range of Fb and Fe demands")

data = []

for Fb in Fb_values:
    for Fe in Fe_values:
        a0.fx[2] = float(Fb)  # Fb demand
        a0.fx[3] = float(Fe)  # Fe demand
        opt_model.solve()
        _, r = getActive(active_reactor)
        _, s = getActive(active_separator)
        combo = int(r) * 10 + int(s)
        print(f"   Fb dem: {Fb}  Fe dem: {Fe}   Optimal combo: {combo}")
        data.append([Fe, Fb, combo])

df = pd.DataFrame(data, columns=["Fb", "Fe", "Combo"])
df_pivot = df.pivot(index="Fe", columns="Fb", values="Combo")

m.close()

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the plot
plt.figure(figsize=(10, 6))
ax = sns.heatmap(df_pivot, annot=True, fmt="", cmap="coolwarm", cbar=False)
plt.title("Optimal Reactor & Separator Selection Heatmap")
plt.xlabel("Fb Demand")
plt.ylabel("Fe Demand")
plt.xticks(rotation=0)
plt.yticks(rotation=0)

print("And finally the heatmap . . .")
plt.show()
