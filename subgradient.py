import gurobipy as gp
from gurobipy import *  # type:ignore
from dataclasses import dataclass
import re
import numpy as np
import copy


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
    for _ in range(N):
        varx_temp = [model.addVar(lb=0, ub=data.demand[j], obj=0, vtype=GRB.INTEGER) for j in range(N)]
        var_x.append(varx_temp)  # x[i][j]从工厂i到客户j的配送量
        var_y.append(model.addVar(lb=0, ub=1, obj=0, vtype=GRB.BINARY))  # y[i] 工厂i是否被选中
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
# model.optimize()

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
print(f"LB: {rmodel.ObjVal}")
#
for i in range(data.cumstomer_num):
    UB += max(data.cost[i])
print(f"UB: {UB}")
