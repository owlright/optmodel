import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import re
import itertools

random.seed(34324)


def print_error(mesg: str):
    from colorama import Fore, Back, Style

    print(Fore.RED + mesg + Style.RESET_ALL)


def print_info(mesg: str):
    from colorama import Fore, Back, Style

    print(Fore.GREEN + mesg + Style.RESET_ALL)


B = [chr(i) for i in range(ord("a"), ord("z") + 1)]
A = [str(i) for i in range(len(B))]
G = nx.Graph()
G.add_nodes_from(A, bipartite=0)
G.add_nodes_from(B, bipartite=1)
G.add_edges_from([(u, v) for u in A for v in B])
# B.add_edges_from([(v, u, {"weight": random.randint(1, 100)}) for u in numbers for v in englist]);
assert nx.is_bipartite(G)

weights = {e: random.randint(10, 100) for e in G.edges()}
arcs = list(weights.keys())
# 关闭gurobi输出license
env = gp.Env(params={"OutputFlag": 0})

m = gp.Model("matching", env=env)
# m.setParam("OutputFlag", 0)
x = m.addVars(arcs, obj=weights, vtype=GRB.BINARY, name="x")
m.addConstrs((x.sum(a, "*") == 1 for a in A), name="must_have_a")
m.addConstrs((x.sum("*", b) == 1 for b in B), name="must_have_b")

m.setAttr("ModelSense", GRB.MINIMIZE)
m.update()
# m.optimize()
m.write(f"{m.ModelName}.lp")

# if m.Status == GRB.OPTIMAL:
# print_info(f"Orignal Model OPT: {m.ObjVal}")

# ! Primal Model
# 对于这个问题直接松弛不影响最优解
P = m.copy().relax()
P.optimize()
P.write(f"relaxed_{m.ModelName}.lp")
primal_objVal = P.ObjVal
print("Primal Obj: ", primal_objVal)


# ! 对偶模型
print(f"Trere are {P.NumConstrs} dual varibles.")
DP = gp.Model("dual_matching", env=env)
DP.setAttr(GRB.Attr.ModelSense, GRB.MAXIMIZE)

ua = DP.addVars(A, obj=1, name="u")
ub = DP.addVars(B, obj=1, name="u")
DP.addConstrs((ua[e[0]] + ub[e[1]] <= weights[e] for e in arcs))
DP.update()
DP.write(f"{DP.ModelName}.lp")
DP.optimize()
print("-------Optimal dual solutions--------")
for var in DP.getVars():
    if var.X > 0:
        print(var.VarName, var.X)
print("-" * 50)


def is_solution_feasible(model: gp.Model, x: dict):
    model.update()
    model = model.copy()
    for k, v in x.items():
        var = model.getVarByName(f"u[{k}]")
        model.addConstr(var == v)
    # model.setObjective(0, sense=GRB.MAXIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        # print("Dual Obj: ", model.ObjVal)
        return True
    elif model.Status == GRB.INFEASIBLE:
        return False
    elif model.Status == GRB.INF_OR_UNBD:
        return False
    else:
        raise RuntimeError("model status is not considered")


# ! Restricted Primal Model
# 设计新的基于对偶松弛约束的主对偶模型
def build_RP(J: list[tuple[str, str]]):
    RP = gp.Model("restricted_model", env=env)
    x = RP.addVars(arcs, name="x")
    sa = RP.addVars(A, lb=0, obj=1, name="s")
    sb = RP.addVars(B, lb=0, obj=1, name="s")
    # 这两个约束的目的只是为了提取出松弛变量sa，sb
    RP.addConstrs((x.sum(a, "*") + sa[a] == 1 for a in A), name="restricted_slack_a")
    RP.addConstrs((x.sum("*", b) + sb[b] == 1 for b in B), name="restricted_slack_b")
    RP.addConstrs((x[e] == 0 for e in set(arcs) - set(J)), name="E-J")  # 对偶模型的松约束，主模型对应变量一定为0
    RP.addConstrs((x[e] >= 0 for e in J), name="J")  # 对偶模型的紧约束对于主模型没有影响，满足主模型自身约束即可
    RP.update()
    RP.write(f"{RP.ModelName}.lp")
    return RP


# 假设我们对于对偶模型通过某种方式获得了初始解，本例中为0，此时DP最优值也为0，接下来将利用RPM不断提升DP的最优值。
ua = {a: 0.0 for a in A}
ub = {b: 0.0 for b in B}
J = [e for e in arcs if ua[e[0]] + ub[e[1]] == weights[e]]
RP = build_RP(J)
RP.optimize()

for _ in range(10):  # this means that P and DP are not optimal
    print(RP.ObjVal)
    # print(sum(ua.values())+sum(ub.values()))

    # ! update dual variables, something wrong here
    # ---------------------------------------------------------------------------- #
    epsilon = min([(weights[e] - ua[e[0]] - ub[e[1]]) for e in set(arcs) - set(J)])  # 挑出还可以提升性能的约束
    for a in A:
        uconstr = RP.getConstrByName(f"restricted_slack_a[{a}]")
        assert uconstr
        ua[a] = ua[a] + epsilon * uconstr.Pi

    for b in B:
        uconstr = RP.getConstrByName(f"restricted_slack_b[{b}]")
        assert uconstr
        ub[b] = ub[b] + epsilon * uconstr.Pi
    # ---------------------------------------------------------------------------- #
    assert is_solution_feasible(DP, {**ua, **ub})
    J = [e for e in arcs if ua[e[0]] + ub[e[1]] == weights[e]]
    RP = build_RP(J)
    RP.optimize()
