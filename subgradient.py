import gurobipy as gp
from gurobipy import *  # type:ignore
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt


@dataclass
class LTPData:
    facility_num_limit: int
    cumstomer_num: int
    supply: list[int]
    demand: list[float]
    cost = []

    def __init__(self, data_path: str):
        with open(data_path) as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                if line[-1] == "\n":
                    line = line[:-1]
                if len(line) < 1:
                    break
                if cnt == 0:
                    self.facility_num_limit = int(line)
                elif cnt == 1:
                    self.cumstomer_num = int(line)
                elif cnt == 2:
                    array = re.split(r" +", line)
                    assert len(array) == self.cumstomer_num
                    self.supply = [int(i) for i in array]
                elif cnt == 3:
                    array = re.split(r" +", line)
                    assert len(array) == self.cumstomer_num
                    self.demand = [float(f) for f in re.split(r" +", line)]
                else:
                    array = re.split(r" +", line)
                    temp = [float(array[j]) for j in range(self.cumstomer_num)]
                    self.cost.append(temp)
                cnt += 1


def create_model(data: LTPData, var_x: list, var_y: list, relaxed_cons: list) -> gp.Model:
    model = gp.Model("Location Transport Problem")
    model.setParam("OutputFlag", 0)
    N = data.cumstomer_num
    for i in range(N):
        varx_temp = [model.addVar(name=f"x[{i}][{j}]",lb=0, ub=data.demand[j], obj=0, vtype=GRB.INTEGER) for j in range(N)]
        var_x.append(varx_temp)  # x[i][j]从工厂i到客户j的配送量
        var_y.append(model.addVar(name=f"y[{i}]",lb=0, ub=1, obj=0, vtype=GRB.BINARY))  # y[i] 工厂i是否被选中
    # 如果工厂i被选中，配送量不能超过其能提供的量
    for i in range(N):
        model.addConstr(quicksum(var_x[i][j] for j in range(N)) <= data.supply[i] * var_y[i])
    # 客户j的需求必须要被满足
    for j in range(N):
        relaxed_cons.append(model.addConstr(quicksum(var_x[i][j] for i in range(N)) >= data.demand[j]))
    # 工厂数量上限
    model.addConstr(quicksum(var_y[i] for i in range(N)) <= data.facility_num_limit)

    model.setObjective(quicksum(var_x[i][j] * data.cost[i][j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
    return model


data = LTPData("location_transport_instance.txt")
max_iter = 200
nochange_cnt_limit = 5
step_size_log = []
theta_log = []
lb_log = []
ub_log = []

x = []
y = []
relaxed_cons = []  # relaxed constraints
model = create_model(data, x, y, relaxed_cons)
assert model
model.update()  # ! must update this model if you don't call optimize
model.optimize()
print(model.ObjVal)
# model.write("original_model.lp")
no_change_cnt = 0
square_sum = 0
step_size = 0
theta = 2.0
LB = 0.0
UB = 0.0
lag_multipler = [0.0] * data.cumstomer_num
slack = [0.0] * data.cumstomer_num

# 松弛模型的整数约束，获得问题的一个下界
model_copy = model.copy()

rmodel = model_copy.relax()
rmodel.optimize()
# rmodel.write("relaxed_model.lp")
objs = []

print(f"LB: {rmodel.ObjVal}")
# 获得问题的上界
for i in range(data.cumstomer_num):
    UB += max(data.cost[i])
print(f"UB: {UB}")
obj_total_cost = quicksum(
    x[i][j] * data.cost[i][j] for i in range(data.cumstomer_num) for j in range(data.cumstomer_num)
)
is_lagrangian_relaxed = False

for iter in range(max_iter):
    if not is_lagrangian_relaxed:
        is_lagrangian_relaxed = True
        for cons in relaxed_cons:
            model.remove(cons)
        relaxed_cons = []

    # lagrangian relaxation term
    lagrangian_relaxed_term = quicksum(
        lag_multipler[j] * (
            data.demand[j] - quicksum(x[i][j] for i in range(data.cumstomer_num))
            )
        for j in range(data.cumstomer_num)
    )
    model.setObjective(obj_total_cost + lagrangian_relaxed_term, GRB.MINIMIZE)
    model.update()
    # model.write("test.lp")
    model.optimize()
    print("obj=", model.ObjVal) # 第一次移除了所有满足用户需求的约束，且slack为0，则目标值应该为0
    objs.append(model.ObjVal)
    for j in range(data.cumstomer_num):
        slack[j] = sum(x[i][j].x for i in range(data.cumstomer_num)) - data.demand[j]
    print(slack)

    # update lower bound if there has any improvement
    if model.ObjVal > LB + 1e-6:
        LB = model.ObjVal
        no_change_cnt = 0
    else:
        no_change_cnt += 1

    # update scale theta if theta does not change for nochange_cnt_limit iterations
    if no_change_cnt == nochange_cnt_limit:
        theta = theta / 2.0
        no_change_cnt = 0
    square_sum = sum([slack[i]**2 for i in range(data.cumstomer_num)])
    # update step size
    step_size = theta*(UB - model.ObjVal) / square_sum
    # update lagrangian multipliers
    for i in range(data.cumstomer_num):
        lag_multipler[i] = lag_multipler[i] - step_size * slack[i]
    else:
        lag_multipler[i] = 0

    selected_facility_supply = sum(data.supply[i] * y[i].x for i in range(data.cumstomer_num))
    demand_sum_all = sum(data.demand)
    print("selected_facility_supply=", selected_facility_supply)
    print("demand_sum_all=",demand_sum_all)

    if selected_facility_supply - demand_sum_all >= 1e-6:
        is_lagrangian_relaxed = False

        for j in range(data.cumstomer_num):
            relaxed_cons.append(model.addConstr(
                quicksum(x[i][j] for i in range(data.cumstomer_num)) >= data.demand[j]
            ))
        # fix y, solve more easily
        for i in range(data.cumstomer_num):
            y[i].lb = y[i].x
            y[i].ub = y[i].x
        model.setObjective(obj_total_cost, GRB.MINIMIZE)
        model.update()
        model.optimize()
        UB = min(UB, model.ObjVal)
        # restore y
        for i in range(data.cumstomer_num):
            y[i].lb = 0.0
            y[i].ub = 1.0
        model.update()
    lb_log.append(LB)
    ub_log.append(UB)
    step_size_log.append(step_size)
    theta_log.append(theta)

plt.rcParams["font.family"] = "Serif"
fig, ax = plt.subplots()#figsize=(15 / 2.54, 8 / 2.54)
fig.subplots_adjust(left=0.10, bottom=0.2, right=0.99, top=0.99)
ax.plot(list(range(len(objs))), objs, marker='o', color="black", markersize=2)
plt.savefig("obj.png", dpi=600)


print("\n    -------------- Iteration log information --------------    \n")
print("  Iter               LB               UB         theta        stepSize")

for i in range(len(lb_log)):
    print("   %3d     %12.6f      %12.6f      %8.6f       %8.6f" % (i, lb_log[i], ub_log[i], theta_log[i], step_size_log[i]))