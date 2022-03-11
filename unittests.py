# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import csv
import os
import pandas as pd

from numpy.random import exponential as npexponential
from numpy.random import choice as npchoice
from numpy.random import randint 
from random import random as rrandom
from random import choice as rchoice
from math import log as ln
from scipy.stats import pearsonr

import unittest
from gillespie_code import *
from helper_functions import *



class TestCode(unittest.TestCase):
    
    def setUp(self):
        self.G = nx.DiGraph()
        self.G.add_nodes_from([1,2,3,4,5])
        self.G.add_edges_from([(1,2),(2,3),(3,1),(3,4),(1,4)])
        self.nodes = list(self.G.nodes())
        i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times = 0.7,0.4,0.95,0.1,0.1,0.1,False,[2]
        self.runvars = SimulationVariables(self.G,4,False,20,i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times)
        self.A = nx.to_numpy_array(self.G)
        
        self.F = nx.DiGraph()
        self.F.add_nodes_from([0,1,2,3,4,5])
        self.F.add_edges_from([(1,0),(1,3),(3,4),(4,5),(5,3)])

        
    def test_adding(self):
        
        item = 8
        X= [3,1,2]
        X_dict = {3:0,2:2,1:1}
        nX = 3
        X,X_dict,nX = adding(item,X,X_dict,nX)
        self.assertEqual(X,[3,1,2,8])
        self.assertEqual(X_dict,{3:0,2:2,1:1,8:3})
        self.assertEqual(nX,4)
        
        item = 3
        X= []
        X_dict = {}
        nX = 0
        X,X_dict,nX = adding(item,X,X_dict,nX)
        self.assertEqual(X,[3])
        self.assertEqual(X_dict,{3:0})
        self.assertEqual(nX,1)
        
    def test_removing(self):
        
        item,ind = 4,1
        X= [3,4,2]
        X_dict = {3:0,2:2,4:1}
        nX = 3
        X,X_dict,nX = removing(item,ind,X,X_dict,nX)
        self.assertEqual(X,[3,2])
        self.assertEqual(X_dict,{3:0,2:1})
        self.assertEqual(nX,2)
        
        item,ind = 3,0
        X= [3]
        X_dict = {3:0}
        nX = 1
        X,X_dict,nX = removing(item,ind,X,X_dict,nX)
        self.assertEqual(X,[])
        self.assertEqual(X_dict,{})
        self.assertEqual(nX,0)
        
    def test_dict_from_list(self):
        
        lis = [1,4,6,4,1,1]
        di = {1:3,4:2,6:1}
        self.assertEqual(dict_from_list(lis),di)

    
    def test_get_degrees(self):
        
        indegs,outdegs = get_degrees(self.G,self.nodes)
        self.assertEqual(indegs,{1:1,2:1,3:1,4:2,5:0})
        self.assertEqual(outdegs,{1:2,2:1,3:2,4:0,5:0})

        
    def test_average_excess_degree(self):
        
        self.assertEqual(average_excess_degree(self.G), np.mean([1,2,2,0,0]))
        
        
    def test_largest_nodes(self):
        
        eig = {1:0.3,2:0.6,3:0.1,4:0.4,5:0.1}
        t = sum(eig.values())
        nodes_eig,vals_eig,sG = largest_nodes(eig,self.G,percentage = 0.6)
        self.assertEqual(nodes_eig, [2,4])
        self.assertAlmostEqual(vals_eig[0], 0.6/t) #,0.4/t])
        self.assertAlmostEqual(vals_eig[1], 0.4/t)
        self.assertEqual(list(sG.nodes()), [2,4])
            
    ###
    
    def test_samedeg_as(self):
        
        indegs = {0:4  , 1: 8  , 2: 4  }
        outdegs = {0: 5 , 1: 5 , 2:  3 }
        indegdict,outdegdict,corr = samedeg_as(indegs,outdegs)
        self.assertEqual(indegdict, {4: 4.0, 8:5.0})
        self.assertEqual(outdegdict, {5:6.0, 3:4.0})
        c,p = pearsonr([4,8,4],[5,5,3])
        self.assertEqual(corr, c)
        
    
    def test_ER_graph(self):
        
        
        G,n = ER_graph(2.4,n=100)
        self.assertEqual(list(G.nodes()),list(np.arange(0,100))) #nodes
        self.assertEqual(n,100)   #n,number of nodes
        self.assertEqual(G.number_of_edges(),int(2.4*100)) #number of links

        
    def test_find_scc(self):
        
        D = self.G.copy()
        D.add_edges_from([(4,6),(6,4)])
        size,scc = find_scc(D)
        self.assertEqual(size,3)
        self.assertEqual(set(scc), {1,2,3})
        
        
    def test_leading_left_eig(self):
    
        lam,eig = leading_left_eig(self.G)
        w,v = np.linalg.eig(self.A)
        truelam = max(w)
        self.assertAlmostEqual(lam,truelam)
    
        
        
        
    def test_get_ccdfs(self):
    
        data = [1,1,1,2,5.5,2,1,10]
        vals,ccdfs = get_ccdfs(data)
        arr = np.asarray([1.,2.,5.5,10.])
        ccdfs_c = np.asarray([1.,0.5,0.25,0.125])
        for i in range(len(arr)):
            self.assertEqual(vals[i],arr[i])
            self.assertEqual(ccdfs[i],ccdfs_c[i])
            
            
    def test_choose_nodeinds(self):
        
        i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times = 0.7,0.4,0.95,0.1,0.1,0.1,False,[2]
        obj = SimulationVariables(self.F,5,False,20,i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times)
        obj.rnlist = [2,4,4,3,2,2,1,3,2]
        
        pairs = [[2,4],[4,3],[1,3]]
        for i in range(4):
            node1,node2 = obj.choose_nodeinds()
            if i < 3:
                self.assertEqual([node1,node2],pairs[i])

    
    def test_initialize_firing(self):
        
        i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times = 0.7,0.4,0.95,0.1,0.1,0.1,False,[2]
        g = epsilon*l
        obj = SimulationVariables(self.F,6,False,20,i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times)
        
        ###initialize_firing
        obj.initialize_firing([1,4])
        self.assertEqual(obj.nF,2)
        self.assertEqual(obj.F, [1,4])
        self.assertEqual(obj.F_dict,{1:0,4:1})

        self.assertEqual(obj.nI,4)
        self.assertEqual(obj.I, [0,2,3,5])
        self.assertEqual(obj.I_dict,{0:0,2:1,3:2,5:3})

        self.assertEqual(obj.nFI,3)
        self.assertEqual(obj.FI, [(1,0),(1,3),(4,5)])
        self.assertEqual(obj.FI_dict,{(1,0):0,(1,3):1,(4,5):2})
        

        ###
        obj.initialize_n_transitions()
        lis = [obj.n_f_to_r,obj.n_i_to_f, obj.n_s, obj.n_l, obj.n_r_to_i, obj.n_g]
        vals = [2*f_to_r, 3*i_to_f, 4*obj.spont_rate, 2*obj.l, 0, obj.g*6]
        for i in range(len(lis)):
            #print(lis[i],vals[i])
            self.assertEqual(lis[i],vals[i])
            
        obj.t_increment = 1.5
        obj.update_nF_sums(1.5)
         
        ####
        #change_F_to_R(self,node):
        node = 1
        obj.change_F_to_R(1)
        self.assertEqual(obj.nF,1)
        self.assertEqual(obj.nR,1)
        self.assertEqual(obj.nFI,1)
        self.assertEqual(obj.F, [4])
        self.assertEqual(obj.F_dict, {4:0})
        self.assertEqual(obj.R, [1])
        self.assertEqual(obj.R_dict, {1:0})
        self.assertEqual(obj.FI, [(4,5)]) 
        self.assertEqual(obj.FI_dict, {(4,5):0})
        
        obj.t_increment = 0.4
        obj.update_nF_sums(1.9)
        
        ###HSP lose
        obj.HSP_lose(3,4,1.9) #at time 1.9
        #check edges
        edges = {(1,0),(1,3),(4,5),(5,3)}
        for edge in obj.G.edges():
            edge in edges == True
        self.assertEqual(obj.n_links,4)
        
        obj.t_increment = 0.7
        obj.update_nF_sums(2.6)
        
        ###change_I_to_F
        obj.change_I_to_F(5)
        self.assertEqual(obj.nF,2)
        self.assertEqual(obj.F, [4,5])
        self.assertEqual(obj.F_dict, {4:0,5:1})
        
        self.assertEqual(obj.nI,3)
        self.assertEqual(obj.I, [0,2,3])
        self.assertEqual(obj.I_dict,{0:0,2:1,3:2})
        
        self.assertEqual(obj.FI, [(5,3)]) 
        self.assertEqual(obj.FI_dict, {(5,3):0})
        self.assertEqual(obj.nFI,1)
        
        obj.t_increment = 0.6
        obj.update_nF_sums(3.2)
        
        ###HSP_add
        obj.HSP_add(5,2,3.2)
        edges = {(1,0),(1,3),(4,5),(5,3),(5,2)}
        for edge in obj.G.edges():
            edge in edges == True
        self.assertEqual(obj.n_links,5)
        
        self.assertEqual(obj.FI, [(5,3),(5,2)]) 
        self.assertEqual(obj.FI_dict, {(5,3):0,(5,2):1})
        self.assertEqual(obj.nFI,2)
        
        obj.t_increment = 0.3
        obj.update_nF_sums(3.5)
        obj.HSP_add(5,1,3.5) #second add
        
        obj.t_increment = 0.5
        obj.update_nF_sums(4.0)
        
        ###change_R_to_I
        obj.change_R_to_I(1)
        
        self.assertEqual(obj.nR,0)
        self.assertEqual(obj.R, [])
        self.assertEqual(obj.R_dict,{})
        
        self.assertEqual(obj.FI, [(5,3),(5,2),(5,1)]) 
        self.assertEqual(obj.FI_dict, {(5,3):0,(5,2):1,(5,1):2})
        self.assertEqual(obj.nFI,3)
        
        self.assertEqual(obj.modified_edges_ts,[1.9,3.2,3.5])
        self.assertEqual(obj.modified_edges_types,[-1,1,1])
        self.assertEqual(obj.modified_edges,[(3,4),(5,2),(5,1)])
        
        ###a
        t_max = 10
        obj.average_activity(t_max)
        val = (3+1.1+2*1.4)/10/6
        self.assertEqual(obj.A,val)


if __name__ == '__main__':
    unittest.main()

