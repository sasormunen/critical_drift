# IFRI-model
Here you can find the code for simulating the directed adaptive IFRI-model described in the paper *Critical drift in a neuro-inspired adaptive network* (Sormunen, Gross & Saramäki, ArXiv: https://doi.org/10.48550/arXiv.2206.10315).  When the adaptation rules are turned off, the model corresponds to the standard SIRS model in epidemiological studies; in short, firing (F) nodes activate their inactive neighbors stochastically at a constant rate, and then enter a refractory (R) period before transitioning back to inactive (I). When the adaptation rules are on, firing nodes lose links at rate *l* while new links are created between random nodes at rate *g*. 

The dynamics and topology are evolved with the Gillespie algorithm. The implementation uses the *networkx* package.

The model can be simulated by calling the function *gillespie(...)* from file *gillespie_code.py*. The parameter options are listed in the file. See the Jupyter notebook *example.ipynb* for a short example. 

Unittests can be run by directly running the script *unittests.py*.
