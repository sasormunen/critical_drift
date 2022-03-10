# IFRI-model
Here you can find the code for simulating the directed adaptive IFRI-model described in the paper *Critical drift in a neuro-inspired adaptive network*.  When the adaptation rules are turned off, the model corresponds to the standard SIRS-model in epidemiological studies; in short, firing (F) nodes activate their inactive neighbors stochastically at rate $\beta$, and then enter a refractory (R) period at rate $\delta$ before transitioning back to inactive (I) at rate $\gamma$. In addition, iactive nodes can fire spontaneously at rate $\eta$. When the adaptatation rules are on, firing nodes lose links at rate *l* while new links are created between random nodes at rate *g*. 

The dynamics and topology are evolved with the Gillespie algorithm. The implementation uses the *networkx* package.

## Example of use

The model can be simulated by calling the function *gillespie(...)* from file *gillespie_code.py*. The parameter options are listed in the file. 

G, avg_deg, rv = gillespie()...
returns the graph, realized average degree and a class instance which stores parameters. 

If HSP==True:
tvals
Means
Lams
Corps
Excessdegs
Deduct
scc_kmeans
scc_sizes
scc_edges

testFs

self.A
self.A_transient
self.comp_size
self.dur


The file *running_functions.py*...

Unittests can be run by directly running the script *unittests.py*.
