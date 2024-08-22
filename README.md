# The Mary Lister McCammon Summer Research Project

This repository contains Julia code and data associated to my Mary Lister McCammon Summer Research Project. In this project I worked on the implementation of a survival regression model based on ordinary differential equations in the [Julia programming language](https://julialang.org/). The main reference for the implemented models is:

> Christen, J. A., and Rubio, F. J. (2024+). Dynamic Survival Analysis: Modelling the Hazard Function via Ordinary Differential Equations. *Statistical Methods in Medical Research*, in press. https://doi.org/10.1177/09622802241268504

The implementation of such models require the use of ODE Solvers ([`DifferentialEquations.jl`](https://docs.sciml.ai/DiffEqDocs/stable/)), as well as MCMC methods from [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

The repository contains the following codes:

1. `rotterdam.jl`: Julia code for ...
2. `routinesODESurv.jl` contains all the functions used in `rotterdam.jl`. 
3. `rotterdamFull.csv` is the data set that contains information of breast cancer patients data for analysis. 
4. `MLE.txt` and `logMLE.txt` are the maximum likelihood estimates of the parameters ($\lambda, \kappa, \alpha, \beta$) in original and logarithm scales respectively.
5. `postsamp.csv` is a file of posterior samples using turing in `rotterdam.jl`. 
6. 

The 

See also [ODESurv](https://github.com/FJRubio67/ODESurv)

# The Mary Lister McCammon Summer Research Fellowship

This fellowship programme is a 10 week research project from 1 July 2024 to 6 September 2024, supported by the Engineering and Physical Sciences Research Council
and by the Department of Mathematics at Imperial College London and University College London. I would like to express deep gratitude to my supervisor Dr Javier Rubio from Department of Statistics at University College London. His guidance and assistance have made this a fruitful, inspiring, and unforgettable experience.

More information about the fellowship can be found at: [Link](https://www.imperial.ac.uk/mathematics/postgraduate/the-mary-lister-mccammon-summer-research-fellowship/)

