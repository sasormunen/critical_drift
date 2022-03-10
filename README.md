# critical_drift
Here you can find the code for the paper "Critical drift in a neuro-inspired adaptive network".
The main function "gillespie" in file "gillespie_code.py" simulates the adaptive IFRI-model described in the paper. If the adaptation rules are turned off (by setting "HSP=False"), the model corresponds to the standard SIRS-model in epidemiological studies. The adaptation rules that firing/infected nodes lose links at rate l while new links are created between random nodes at rate g. 


The dynamics and topology are evolved with the Gillespie algorithm. The implementation uses the networkx package.

## Example of use

by calling the function "gillespie()" from file gillespie_code.py. The parameter options are listed in the file. 
G, avg_deg, rv = gillespie()...
returns the graph, realized average degree and a class instance. 

Unittests can be run by directly running the script unittests.py.
