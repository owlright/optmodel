import networkx as nx
import random
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import re
import itertools as itt
random.seed(1234)
g = nx.random_graphs.random_regular_graph(3, 10).to_directed()
weights = {e: random.randint(1, 9) for e in g.edges()}
arcs = list(weights.keys())
N = list(g.nodes)
S = [4, 7, 0, 1, 9]
r = 8
V = list(set(N) - set(S))
V.remove(r)
print(S, V, r)

stp = gp.Model("netflow")

x = stp.addVars(S, arcs, lb=0, vtype=GRB.INTEGER, name="x")
y = stp.addVars(arcs, obj=weights, vtype=GRB.BINARY, name="y")
for k in S:
    for i in N:
        volume = 0
        if i == r:
            volume = 1
        elif i == k:
            volume = -1
        stp.addConstr(x.sum(k, i,'*') - x.sum(k,'*', i) == volume, name="flow_conversation")
stp.addConstrs((x[k, i, j] <= y[i, j] for i, j in arcs for k in S), name="choose_edge")

stp.setAttr("ModelSense", GRB.MINIMIZE)
stp.optimize()
stp.write("stp.lp")
variable_names = [var.VarName for var in stp.getVars()]
flows = {s:[] for s in S}
for v in stp.getVars():
    if (v.X > 0):
        matches = re.findall(r'\d+', v.VarName)
        if len(matches) == 3:
            k, i, j = (int(match) for match in matches)
            flows[k].append((i,j))
        else:
            print(v.VarName)
print(flows)
pos = nx.kamada_kawai_layout(g)
nx.set_node_attributes(g, pos, 'pos')
nx.set_edge_attributes(g, weights, "cost")
nx.draw(g, pos=pos, with_labels=True)
plt.show()
