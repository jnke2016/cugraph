import networkx as nx
import cugraph 
import pandas as pd
import numpy as np
import cudf
from os import walk



#_, _, filenames = next(walk("datasets_2"))
#t="datasets_2/"+(filenames[0])
#y=[t]
#print (d)

#Dataset_drctd=["datasets/p2p-Gnutella04.txt", "datasets/p2p-Gnutella05.txt", "datasets/wiki-Vote.txt", "datasets/email-EuAll.txt"]
Dataset_drctd=["datasets/email-EuAll.txt"]
Dataset_undrctd_1=["datasets/karate_mod.mtx"]
Dataset_bprtid=["datasets/wikiElec.ElecBs3.txt"]
Dataset_undrctd=["datasets/ca-AstroPh.txt", "datasets/ca-CondMat.txt", "datasets/ca-GrQc.txt", "datasets/ca-HepTh.txt", "datasets/facebook_combined.txt", "datasets/email-Enron.txt"]


for d in Dataset_undrctd_1:
    #d = "datasets_2/"+ str(d)
    g = nx.read_edgelist(d,create_using=nx.Graph(), nodetype = float)
    #print(nx.info(g))
    print("graph ",d,"\nNodes with self loops\n",list(nx.nodes_with_selfloops(g)))
    print("The number of edges is ",g.number_of_edges())
    print("The number of nodes is ",g.number_of_nodes())
    print("\n\n")
   
    #print(list(nx.selfloop_edges(g)))
    #print("\n")
    #print("isolated vertex \n",list(nx.isolates(g)))
    print("\n\n")


#check the betweenness centrality both with cugraph and netowrkX

#M_nx = pd.read_csv('datasets/karate_mod.mtx', delimiter=' ',
#                   header=None)
#print(M_nx)

"""
G_nx=nx.from_pandas_edgelist(M_nx, source=0, target=1) #make sure the datatype matches. '0' and '1' is wrong
bc_nx = nx.betweenness_centrality(G_nx)
print(bc_nx)
"""

"""
G = cugraph.Graph()
G.from_pandas_edgelist(M_nx, source=0, destination=1)
bc = cugraph.betweenness_centrality(G)
print(bc)
"""



dataset="datasets/karate_str.mtx"
g=cudf.read_csv(dataset,delimiter=" ")
#print(g)
