#######################################################################################
# Data preparation
#######################################################################################

rm(list=ls())

# Required packages
library(deSolve)
library(survival)
library(ggplot2)
#library(devtools)
#install_github("FJRubio67/HazReg")
library(HazReg)
library(Rtwalk)
library(spBayes)
library(knitr)
library(demography)


## ----include=FALSE----------------------------------------------------------------------------------------

#source("C:/Users/Javier/Dropbox/ODESurv/Codes/routines/routines.R")
source("~/Dropbox/ODESurv/Codes/routines/routines.R")


## ----eval=FALSE-------------------------------------------------------------------------------------------
## source("routines.R")


## ---------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Data preparation
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

# Parameter values
lambda <- 1.8
kappa <- 0.1
alpha <- 6
beta <- 4.8
h0 <- 1e-2
q0 <- 1e-6
 
truepar <- c(lambda, kappa, alpha, beta)

# Sample size
n <- 250

# Censoring time
cens <- 20

# Simulation specifications
nsim <- 250 # number of Monte Carlo simulations
NMC <- 55000 # number of MCMC iterations
burn <- 5000 # burn-in period
thin <- 50 # thinning period

# Posterior sample after burn-in and thinning
ind=seq(burn,NMC,thin) 

postHT <- list()




for(j in 1:nsim){
  print(j)
  
  # Data simulation
set.seed(j)
sim <- simHT(n = n, par = truepar, tmin = 0, tmax = 150, step = 1e-3)$sim

status <- ifelse(sim < cens, 1, 0)

time <- ifelse(sim < cens, sim, cens)

# New data frame: logical status, time in years, survival times sorted
df <- data.frame(time = time, status = status)
df$status <- as.logical(df$status)


df <- df[order(df$time),]

# Required quantities
status <- as.logical(df$status)
t_obs <- df$time[status]
survtimes <- df$time

#==================================================================================================
# Bayesian Analysis
#==================================================================================================

# Initial point

# Optimisation step
OPTHT <- nlminb(log(truepar[1:4]) , log_likHTL, control = list(iter.max = 10000))

#--------------------------------------------------------------------------------------------------
# Hazard-Treatment ODE model for the hazard function: Solver solution
#--------------------------------------------------------------------------------------------------


n.batch <- 1100
batch.length <- 50

lp <- function(par) -log_postHTL(par)

inits <- OPTHT$par

set.seed(j)
infoHT <- adaptMetropGibbs(ltd=lp, starting=inits, accept.rate=0.44, batch=n.batch, 
                           batch.length=batch.length, report=100, verbose=FALSE)

chainHT <- infoHT$p.theta.samples[,1:4]

# Burning and thinning the chain
burn <- 5000
thin <- 50
NS <- n.batch*batch.length
ind <- seq(burn,NS,thin)

postHT[[j]] <- exp(chainHT[ind,1:4])

}

#setwd("~/Dropbox/ODESurv/Codes/Simulations/HT/cens50/n250")
setwd("~/Dropbox/ODESurv/Codes/Simulations/HT/cens25/n250")
save.image("HTn250c25.RData")
