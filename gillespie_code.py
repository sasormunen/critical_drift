# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import time
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

from helper_functions import *
import unittest



def removing(item,ind,X,X_dict,nX):
    
    last = X[-1]
    X[ind] = last
    X.pop()
        
    X_dict[last] = ind
    del X_dict[item]
    nX -= 1
    
    return X,X_dict,nX


def adding(item, X,X_dict,nX):
    
    X.append(item)
    X_dict[item] = nX
    nX += 1
    
    return X,X_dict,nX


class SimulationVariables:

    """ used in the gillespie-simulation """

    def __init__(self,G,n,HSP,transient,i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times):
        
        self.G = G
        self.n_links = self.G.number_of_edges()
        self.nodes = list(self.G.nodes)
        self.n = n
        self.avgdeg_realized = self.n_links / self.n
        
        self.transient = transient
        self.i_to_f = i_to_f
        self.r_to_i = r_to_i
        self.f_to_r = f_to_r
        self.l = l
        self.epsilon = epsilon
        self.spont_rate = spont_rate
        self.dies_when_dies = False #True if HSP == False and spontaneous_firing == False
        self.degs_times = degs_times
        if spontaneous_firing == False:
            self.spont_rate = 0
            if HSP == False:
                self.dies_when_dies = True
        if HSP == True:
            self.g = epsilon*l #setting the rate at which nodes gain links
        else:
            self.g,self.l = 0,0
            
        self.F,self.I,self.R,self.FI = ([] for i in range(4))
        self.F_dict,self.I_dict,self.R_dict,self.FI_dict = ({} for i in range(4))
        self.nF=self.nI=self.nR=self.nFI = 0
        
        self.nF_sum = 0
        self.transient_sum = 0
        self.comp_size = 1
    
        self.tvals = []
        self.kmeans = []
        self.lams = []
        self.corrs = []
        self.excessdegs = []
        self.degdict = {}
        self.scc_sizes = []
        self.scc_edges = []
        
        
        self.testFs = {}
        self.activation_times = {}
        for node in self.nodes:
            self.testFs[node] = 0
            self.activation_times[node] =0
            
        self.modified_edges_ts = []
        self.modified_edges = []
        self.modified_edges_types = []
        
        self.rnlist = randint(0,n,10**5) #faster generation of random numbers
        self.rn_ind = 0   
        
        self.timeseries_F = []
        self.timeseries_t = []
            
    def average_activity(self,t_max):
        
        avg_fs = self.nF_sum / t_max #getting the integral – nF_sum is the area
        self.A = avg_fs / self.n #to a fraction
        after_transient = (self.nF_sum - self.transient_sum)/(t_max-self.transient_t)
        self.A_transient = (after_transient/self.n)

        
    def choose_nodeinds(self):
        
        while True:
            try:
                ind1,ind2 = self.rnlist[self.rn_ind],self.rnlist[self.rn_ind+1]
            except: #if almost all the RNs in the list have already been used
                self.rnlist = randint(0,self.n,10**5) 
                self.rn_ind = 0
                ind1,ind2 = self.rnlist[self.rn_ind],self.rnlist[self.rn_ind+1]
            self.rn_ind += 2
            node1,node2 = self.nodes[ind1],self.nodes[ind2]
            if node1 != node2:
                break
                
        return node1,node2
            
    ###
    
    def initialize_firing(self,firing):
        
        inactives = set(self.nodes) - set(firing)
    
        for fnode in firing:
            self.F.append(fnode)
            self.F_dict[fnode] = self.nF
            self.nF += 1

        for node in inactives:
            self.I.append(node)
            self.I_dict[node] = self.nI
            self.nI += 1

        for node in firing:
            for successor in self.G.successors(node):
                if successor not in self.F_dict: #this is faster since len(I) > len(F)
                    self.FI.append((node,successor))
                    self.FI_dict[(node,successor)] = self.nFI #key = link, val = index in the FI_list
                    self.nFI += 1
                    
    
                    
    
    def change_F_to_R(self,node):
    
        ind = self.F_dict[node]
        _,_,self.nF = removing(node,ind,self.F,self.F_dict,self.nF)
        _,_,self.nR = adding(node,self.R,self.R_dict,self.nR)


        for successor in self.G.successors(node):
            if successor in self.I_dict:
                pair = (node,successor)
                ind = self.FI_dict[pair]
                _,_,self.nFI = removing(pair,ind,self.FI,self.FI_dict,self.nFI)
                
        self.n_f_to_r = self.nF*self.f_to_r 
        self.n_r_to_i = self.nR*self.r_to_i 
        self.n_i_to_f = self.nFI*self.i_to_f
        self.n_l = self.nF*self.l
                
             
    def change_R_to_I(self,node):
        
        ind = self.R_dict[node]
        _,_,self.nR = removing(node,ind,self.R,self.R_dict,self.nR)
        _,_,self.nI = adding(node,self.I,self.I_dict,self.nI)

        for predecessor in self.G.predecessors(node):
            if predecessor in self.F_dict:
                _,_,self.nFI = adding((predecessor,node),self.FI,self.FI_dict,self.nFI)
                
        self.n_r_to_i = self.nR*self.r_to_i #n becoming inactive
        self.n_i_to_f = self.nFI*self.i_to_f    #n becoming firing due to simulation
        self.n_s      = self.nI*self.spont_rate      #n becoming firing spontaneously #n - F_ind = number of I_nodes
        
    
    def change_I_to_F(self,node):
        
        ind = self.I_dict[node]    
        _,_,self.nF =adding(node,self.F,self.F_dict,self.nF)
        _,_,self.nI = removing(node,ind,self.I,self.I_dict,self.nI)

        for successor in self.G.successors(node):
            if successor in self.I_dict:
                _,_,self.nFI = adding((node,successor),self.FI,self.FI_dict,self.nFI)

        #remove from FI any pairs where predecessor of "node" are firing
        for predecessor in self.G.predecessors(node):
            if predecessor in self.F_dict:
                pair = (predecessor,node)
                ind = self.FI_dict[pair]
                _,_,self.nFI = removing(pair,ind,self.FI,self.FI_dict,self.nFI)
                
        self.n_f_to_r = self.nF*self.f_to_r #n becoming refractory   #
        self.n_i_to_f = self.nFI*self.i_to_f      #n becoming firing due to simulation
        self.n_s      = self.nI*self.spont_rate      #n becoming firing spontaneously #n - F_ind = number of I_nodes
        self.n_l = self.l*self.nF
                
    
    def HSP_lose(self,pre,node,t):
        
        self.G.remove_edge(pre,node)
        self.n_links -= 1

        self.modified_edges_ts.append(round(t,1))
        self.modified_edges_types.append(-1)
        self.modified_edges.append((pre,node))
        
    
    def HSP_add(self,node1,node2,t):
    
        self.G.add_edge(node1,node2)
        
        if node1 in self.F_dict and node2 in self.I_dict:
            _,_,self.nFI = adding((node1,node2),self.FI,self.FI_dict,self.nFI)         #update FI

        self.n_links += 1

        self.modified_edges_ts.append(round(t,1))
        self.modified_edges_types.append(1)
        self.modified_edges.append((node1,node2))
        
        self.n_i_to_f = self.nFI*self.i_to_f
        

        
    def initialize_n_transitions(self):
        
        self.n_f_to_r = self.nF*self.f_to_r #n becoming refractory   #
        self.n_i_to_f = self.nFI*self.i_to_f     #n becoming firing due to simulation
        self.n_s      = self.nI*self.spont_rate      #n becoming firing spontaneously #n - F_ind = number of I_nodes
        self.n_l      = self.nF*self.l
        self.n_r_to_i = self.nR*self.r_to_i #n becoming inactive
        self.n_g      = self.g*self.n
    
    def get_n_transitions(self):
        
        self.n_transitions = float(self.n_f_to_r  + self.n_r_to_i + self.n_i_to_f + self.n_s + self.n_l + self.n_g)
    
    def get_probabilities(self):
        
        self.p_f_to_r = self.n_f_to_r / self.n_transitions
        self.p_r_to_i = self.n_r_to_i / self.n_transitions
        self.p_i_to_f = self.n_i_to_f / self.n_transitions
        self.p_s      = self.n_s      / self.n_transitions
        self.p_l      = self.n_l / self.n_transitions
        self.p_g      = self.n_g / self.n_transitions
        
    def new_t(self,t):
        
        self.t_increment = -ln(rrandom()) / self.n_transitions
        t = t + self.t_increment #npexponential(n_transitions)
        return t
    
    def update_nF_sums(self,t):
        
        self.nF_sum += self.t_increment * self.nF #use the next t_increment and the current nF
        if t < self.transient:
            self.transient_sum += self.t_increment * self.nF
            self.transient_t = t #write over so that the last one will be saved
            
    
    def fix_premature_testFs(self,t):
            
        for node in self.F:
            self.testFs[node] += (t - self.activation_times[node])
    
    def track_stuff(self,t,track_SCC,nexttrack,degs_times):
        
        self.tvals.append(nexttrack)
        self.kmeans.append(float(self.n_links)/self.n)

        indegs,outdegs = get_degrees(self.G,self.nodes)
        indeg_means,outdeg_means,corr = samedeg_as(indegs,outdegs)
        self.corrs.append(corr)

        #excess degree
        excessdeg = average_excess_degree(self.G)
        self.excessdegs.append(excessdeg)

        #lambda and eigs
        lam,eig = leading_left_eig(self.G)
        self.lams.append(lam)      #lnodes,lvals,sG = largest_nodes(eig,G,percentage = 0.2)

        if track_SCC == True:
            
            scc_size,scc_nodes = find_scc(self.G)
            self.scc_sizes.append(scc_size/self.n)

            scc_edge = 0
            for edge in self.G.edges():
                if edge[0] in scc_nodes and edge[1] in scc_nodes:
                    scc_edge += 1

            scc_edge = scc_edge / self.G.number_of_edges()
            self.scc_edges.append(scc_edge)
            
        #degree distributions
        if nexttrack in self.degs_times:
            indegs,outdegs = list(indegs.values()),list(outdegs.values())
            indegs = dict_from_list(indegs)
            outdegs = dict_from_list(outdegs)
            print("t",nexttrack)
            print("indegs",indegs)
            print("outdegs",outdegs)     #runvars.degdict[t] = {"indegs":indegs,"outdegs":outdegs}

          


            



def gillespie(graphtype = "ER", i_to_f = 0.7, f_to_r = 0.95, r_to_i = 0.4, t_max = 3*10**5, spont_rate = 10**(-6),n = 10**4,
                  avg_deg = 3, fraction_firing = 0.05, verbose = False,
                  spontaneous_firing = True,
                  HSP = False, l = 0.001, epsilon = 0.01,
                  track_SCC = False, testF = False, 
                  writeout = False,runvars = None,degs_times = [10**6], record_step = 500,transient = 500,writeout_step = 10**6,idval = 0,dir = "../csvfiles/"):
    

    """ simulates the IFRI-model with the gillespie algorithm
    
        inactive (I) -> firing (F) -> refractory (R) -> inactive
                        
        graphtype - "ER" or a ready networkx graph G
        spont_rate - spontaneous firing rate (to balance finite size effects)
        l - determines the rate at which active nodes lose links
        fraction_firing - initialized with int(fraction_firing*n) nodes in the firing state
        HSP - if True, applies the adaptation rules
        t_max - time limit before the simulation is ended
        track_SCC - if True, tracks the strongly connected component
        testF - forms a dict of type {node: #time a node was activated during the run}
        degstops - saves the degree dist at the specified times
        runvars - either None or a SimulationVariables object
        if HSP == True, writes 2 addition files, where first is the edgelist at t=0, and
        the other tracks all edge additions/removals (columns: time, edge, removal/addition (-1 or 1))
        record_step - determines how often the network parameters (mean degree, leading eigenvalue of the adjacency matrix, Pearson correlation coefficient of nodes in-and out-degrees and the mean excesss degree) are calculated
        writeout_step - determines how often the parameter lists are written out to files
        idval - identifier to the filenames in case one needs to do multiple runs with the same parameters
        dir - directory where subdirectories for the csvfiles are created
        """
        
    #creating the graph
    
    if graphtype == "ER":
        G,n = ER_graph(avg_deg,n)
    else:
        G = graphtype
        n = len(list(G.nodes()))
    
    ###
    outfile = ("runvars_startdeg_" + str(avg_deg) + "_nnodes_" + str(n) +
               "_HSP_" + str(HSP) + "_timesteps"+ "_" + str(t_max) + "_spontrate_" + str(spont_rate) +
                "_l_" + str(l) + "_epsilon_" + str(epsilon))
    
    if HSP == True and writeout == True:
        
        for folder in ["edgechanges","kmeans"]: #check that the directories exist:
            if not os.path.exists(dir+folder):
                os.makedirs(dir+folder)
                
        nx.write_edgelist(G,dir+"edgechanges/"+outfile+"_id_"+str(idval)+"_t0.edgelist") #write edgelist
        
        
    ###
    
    if runvars == None:
        rv = SimulationVariables(G,n,HSP,transient,i_to_f,r_to_i,f_to_r,l,epsilon,spont_rate,spontaneous_firing,degs_times)
        
    firing = npchoice(rv.nodes,int(fraction_firing*n), replace = False) #set small fraction firing
    rv.initialize_firing(firing)
    rv.initialize_n_transitions()
    
    if testF == True:
        for node in firing:
            rv.activation_times[node] = 0
    rv.timeseries_F.append(rv.nF/rv.n)
    rv.timeseries_t.append(0)

    
    t = nexttrack = 0
    next_writeout = writeout_step
    start_time = time.time()
    
    if HSP == True and writeout == True:
        kmeansfile ="../csvfiles/kmeans/" + outfile + "_id_"+str(idval)+".csv"
        edgechangesfile ="../csvfiles/edgechanges/"+outfile+"_id_"+str(idval)+ "_edgechanges.csv"
        write_several_rows(kmeansfile,[rv.tvals, rv.kmeans,  rv.lams, rv.corrs,rv.excessdegs])
        write_several_rows(edgechangesfile, [rv.modified_edges_ts,rv.modified_edges,rv.modified_edges_types])
    
    while True:
        
        #######end condition 1
        if rv.dies_when_dies == True and rv.nF == 0: #in this case the run is ended, t still corresponds to the time at which the last transition was made
            rv.finished = True #died before t_max
            break
        #######

        rv.get_n_transitions()
        rv.get_probabilities() #get the probabilites for types of transitions
        t = rv.new_t(t)
        rv.update_nF_sums(t)
            
        
        if HSP == True and t >= nexttrack:   #calculate network parameters to be tracked at desired timepoints
        
            if nexttrack == next_writeout:
                if writeout == True:
                    write_several_rows(kmeansfile,[rv.tvals, rv.kmeans,  rv.lams, rv.corrs,rv.excessdegs],"a")
                    write_several_rows(edgechangesfile, [rv.modified_edges_ts,rv.modified_edges,rv.modified_edges_types],"a")
                    
                rv.tvals,rv.kmeans,rv.lams,rv.corrs,rv.excessdegs = [],[],[],[],[]
                rv.modified_edges_ts,rv.modified_edges,rv.modified_edges_types = [],[],[]
                next_writeout += writeout_step
                
            
            while True: #just in case t_increment would be so large that several track_steps would be passed at once (probably never happens, in case happens, this should be implemented in a smarter way (no double calculations))
                
                rv.track_stuff(t,track_SCC,nexttrack,degs_times)
            
                if verbose == True:
                    print(t,round(nexttrack,3),round(rv.n_links/rv.n,4),"A: ",rv.nF/rv.n, "lam:",round(rv.lams[-1],4), "exc:", round(rv.excessdegs[-1],4)) #"assort: ", round(assort,3), "excess: ", round(excessdeg,3), #,

                nexttrack += record_step
                if t < nexttrack:
                        break
                        
        #######end condition 2
        if t > t_max:
            rv.finished = False
            t = t_max #these should be the same for the sake of plotting
            
            if testF == True:
                rv.fix_premature_testFs(t)
            break
        #######
        
        u = rrandom() #perform one transition depending on their probabilities

        if u < rv.p_f_to_r: #change F to R
            node = rchoice(rv.F)
            rv.change_F_to_R(node)

            if testF == True:
                rv.testFs[node] += (t - rv.activation_times[node])
                rv.timeseries_F.append(rv.nF/rv.n)
                rv.timeseries_t.append(t)
        
        elif u < (rv.p_f_to_r+rv.p_r_to_i): #change refractory to inactive
            node = rchoice(rv.R)
            rv.change_R_to_I(node)
            
        elif u < (rv.p_f_to_r+rv.p_r_to_i+rv.p_i_to_f + rv.p_s): #depending on u, either fire when excited (F-I link) or spontaneously
            
            if u <= (rv.p_f_to_r+rv.p_r_to_i+rv.p_i_to_f): #change some I (from an SI-link) to F
                link = rchoice(rv.FI)
                fnode,node = link[0],link[1]
                rv.comp_size += 1
                     
            else: #spontaneously
                node = rchoice(rv.I)

            rv.change_I_to_F(node)
            if testF == True:
                rv.activation_times[node] = t
                rv.timeseries_F.append(rv.nF/rv.n)
                rv.timeseries_t.append(t)
                  
        elif u < (rv.p_f_to_r+rv.p_r_to_i+rv.p_i_to_f + rv.p_s+rv.p_l): #choose an F-node and then randomly one of its incoming links

            node = rchoice(rv.F)
            pres = list(rv.G.predecessors(node))

            if len(pres) > 0:
                pre = rchoice(pres)
                rv.HSP_lose(pre,node,t)

        else: #add randomly one link (HSP)
            
            node1,node2 = rv.choose_nodeinds()      
            if rv.G.has_edge(node1,node2) == False:
                rv.HSP_add(node1,node2,t)
                

    #after all iterations (if avalanche dies out, this is where we jump
    rv.average_activity(t_max)
    rv.dur = t
    if transient > t_max: #just checking
        print("transient fail")
        
    if verbose == True:
        duration = (time.time() - start_time) #/ float(60)
        print(duration, " s")
    
    if writeout == True and len(rv.tvals) > 0 and HSP == True: #write to csv file
        write_several_rows(kmeansfile,[rv.tvals, rv.kmeans,  rv.lams, rv.corrs,rv.excessdegs],"a")
        write_several_rows(edgechangesfile, [rv.modified_edges_ts,rv.modified_edges,rv.modified_edges_types],"a")
            
    
    return G, rv.avgdeg_realized, rv


