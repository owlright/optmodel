import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

column_names = ["CUST NO", "XCOORD", "YCOORD", "DEMAND", "READY TIME", "DUE DATE", "SERVICE TIME"]
df = pd.read_csv("c101.txt", header=None, skiprows=9, delimiter=r"\s+")
df.columns = column_names
df.drop(columns=["CUST NO"], inplace=True)
print(df)
pos = df[['XCOORD', 'YCOORD']].values
print(pos.shape)
g = nx.Graph(instance="c101")
for i in range(pos.shape[0]):
    g.add_node(i, pos = pos[i, :])
nx.draw(g, pos=nx.get_node_attributes(g, 'pos'))
plt.show()
