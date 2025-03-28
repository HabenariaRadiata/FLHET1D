# Some comments

To run the 1D code, just run `python FLEHT_compiled_1d.py config_BM_charoy.ini`
In the init file, one should choose the initial condition of the discharge parameters from `Input profiles directory`. 

The anomalous transport terms and the out-of-diagonal pressure term
```math
R_{ei}^x, R_{ei}^y, P_{e,xy}
```
are initialized by a txt file from `Empirical term path` in the inputfile.