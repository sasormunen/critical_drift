# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import csv
import os
import pandas as pd

from numpy.random import randint
from scipy.stats import pearsonr




def get_degrees(G,nodes):

    """ returns dictionaries of nodes' in-and out-degrees. G is a networkx-object"""
    
    indegs = {}
    outdegs = {}
    
    for node in nodes:
        indegs[node] = G.in_degree(node)
        outdegs[node] = G.out_degree(node)
        
    return indegs,outdegs
    
    

def samedeg_as(indegs,outdegs):
    
    
    indegdict,outdegdict = {},{}
    for node,indeg in zip(indegs.keys(),indegs.values()):
        outdeg = outdegs[node]
        try:
            indegdict[indeg].append(outdeg)
        except:
            indegdict[indeg] = [outdeg]

        try:
            outdegdict[outdeg].append(indeg)
        except:
            outdegdict[outdeg] = [indeg]
            
    for di in [indegdict,outdegdict]:
        for key in di:
            di[key] = np.mean(di[key])
            
    corr,p = pearsonr(list(indegs.values()),list(outdegs.values()))
        
    return indegdict,outdegdict,corr



def average_excess_degree(G):

    """ average excess degree in networkx-graph G """
    
    degs = []
    for edge in G.edges():
        node = edge[1]
        k = G.out_degree(node)
        degs.append(k)
    
    return np.mean(degs)




def dict_from_list(vals):
    
    """ returns a dict of the form {val : frequency in vals} for val in vals"""
    
    di = {}
    
    for val in vals:
        if val in di:
            di[val] += 1
        else:
            di[val] = 1
            
    return di



def largest_nodes(eig,G,percentage = 0.8):
    
    a = sorted(eig.items(), key=lambda x: x[1])  
    
    nodes,vals = [],[]
    #print(a)
    for pair in a:
        nodes.append(pair[0])
        vals.append(pair[1])
       
    nodes.reverse()
    vals.reverse()
    
    vals = list(np.asarray(vals)/sum(vals))

    #if sum(vals) > 1.02 or sum(vals) < 0.98:
        #print("eigs not summing up to 1",sum(vals))
    
    f,cut_ind = 0,0
    for j in vals:
        f += j
        cut_ind += 1
        if f > percentage:
            break
            
    sG = G.subgraph(nodes[:cut_ind])
    
    return nodes[:cut_ind], vals[:cut_ind], sG



def get_ccdfs(data):
    
    data = np.asarray(data)
    n = len(data)
    unique_vals = np.sort(np.unique(data))
    ccdfs = np.zeros(len(unique_vals))
    
    for i,val in enumerate(unique_vals):
        ccdfs[i] = np.sum(data >= val)/n
    
    return unique_vals,ccdfs


def ER_graph(avg_deg = 2.0,n = 10**4):
    
    """ creates an ER-graph with avg_deg*n links """
    
    n_links = int(avg_deg * n)
    
    G = nx.DiGraph()
    nodes = np.arange(0,n)
    rans = np.random.randint(0,n,2*n_links +500)
    
    G.add_nodes_from(nodes)
    
    i=hits=0
    while hits < n_links:
        n1,n2 = rans[i],rans[i+1]
        if not n1 == n2 and G.has_edge(n1,n2) == False:
            hits += 1
            G.add_edge(n1,n2)
            #if unidirec == True:
            #if not G.has_edge(n2,n1):
                #G.add_edge(n2,n1)
        i += 2

    return G,n






def find_scc(G):
    
    """ returns the size of the largest strongly connected component as well as a list of the nodes forming it """

    scccomps = nx.strongly_connected_components(G)

    lens = []
    scc = []
    for scomp in scccomps:
        ssize = len(list(scomp))
        lens.append(ssize)
        scc.append(list(scomp))
        
    mxind = np.argmax(lens)
    mx,scc =  lens[mxind], scc[mxind]

    return mx,scc



def leading_left_eig(G):
    
    """ returns the leading eigenvalue of G's adjacency matrix and the corresponding principal left eigenvector (dict)"""

    eig = nx.eigenvector_centrality(G,max_iter = 100000) #left eigenvector centrality
    A = nx.adjacency_matrix(G)

    x = np.asarray(list(eig.values()))
    pro = x @  A 
    i= 0
    while True:
        lam = pro[i] / x[i]
        if pro[i] > 0.001:
            break   
        i+= 1
    
    return lam,eig


          

def write_several_rows(outfile,list_of_lists,mode='w'):

    """for example [[1,2,3],[4,5,6],[7,8,9]] is written so that each of the smaller lists becomes one column """

    with open(outfile, mode) as csvFile:
        writer = csv.writer(csvFile, delimiter=' ')
        
        n_iterates = len(list_of_lists)
        lislen = len(list_of_lists[0])
        
        for varind in range(lislen):
            line = []
            for i in range(n_iterates):
                lis = list_of_lists[i]
                line.append(lis[varind])
            writer.writerow(line)




def read_graph(path,N):

    """ creates a graph from edgelist, all nodes from 0 to N_1 are included (even if they have no links """
    
    G = nx.read_edgelist(path, create_using=nx.DiGraph(),nodetype = int)
    
    nodes_now = list(G.nodes())
    nodes = list(np.arange(0,N)) #add the missing nodes (nodes with no links)
    missing = set(nodes) - set(nodes_now)
    for node in missing:
        G.add_node(node)
    
    return G,N
    
    
    
def degfiles_from_dir(fname, path,extra=[""]):

    """ looks up all files in path which start with: fname + "_avgdeg_X_ ....
        and which include at least one of the strings given by extra
        -> {deg: [list of filenames] (not necessarily in the index order)"""

    degfiles = {}
    if fname[-1] != "_":
        fname = fname + "_"

    for fil in os.listdir(path):
        if fil.startswith(fname + "avgdeg"):
            proceed = False
            for e in extra:
                if e in fil:
                    proceed = True
            if proceed:
                deg = fil[fil.index("avgdeg_")+7:]
                deg = deg.split('_')[0]
                if not deg in degfiles:
                    degfiles[deg] = [fil]
                else:
                    degfiles[deg].append(fil)
                
    return degfiles
