# critical_drift
Here you can find the code for the paper "Critical drift in a neuro-inspired adaptive network". 
Simulates the adaptive IFRI-model described in the paper. If the adaptation rules are turned off, the model corresponds to the standard SIRS-model in epidemiological studies. 
The dynamics and topology are evolved with the Gillespie algorithm.

# Example of use

by calling the function "gillespie()" from file gillespie_code. The parameter options are listed in the file. 
G, avg_deg, rv = gillespie()...
returns the graph, realized average degree and a class instance. 

Unittests can be run by directly running the script unittests.py
