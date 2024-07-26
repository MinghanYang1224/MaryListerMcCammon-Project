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
library(bshazard)
library(splines)

## ----include=FALSE----------------------------------------------------------------------------------------

#source("C:/Users/Javier/Dropbox/ODESurv/Codes/routines/routines.R")
source("~/Dropbox/ODESurv/ODESurv/Codes/routines/routines.R")
load("HTn250c25.RData")

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
npar <- length(truepar)

# Sample size
n <- 250

# Censoring time
cens <- 20

# Simulation specifications
nsim <- 250 # number of Monte Carlo simulations

# Time grid
tvec <- seq(0,cens,by = 0.1)
ntvec <- length(tvec)

NMCMC <- nrow(postHT[[1]]) # number of MCMC samples

distbs <- vector()
distp <- vector()

################################################################################################
# True hazard
################################################################################################

# Hazard-Response


paramsHRT  <- c(lambda = truepar[1], kappa = truepar[2], alpha = truepar[3],
                  beta = truepar[4])
initHRT      <- c(h = h0, q = q0, H = 0 )

outT <- ode(initHRT, tvec, hazmodHR, paramsHRT, method = "lsode", jacfunc = jacODE, jactype = "fullusr")
  
hT <- outT[,2]

################################################################################################
# Simulation
################################################################################################


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

#--------------------------------------------------------------------------------------
# B-splines model 
#--------------------------------------------------------------------------------------

fit <- bshazard(Surv(df$time, as.numeric(df$status)) ~ 1, data = df, nbin = ntvec, verbose = FALSE)

#--------------------------------------------------------------------------------------
# Hazard-Response model 
#--------------------------------------------------------------------------------------

hCIHR <- matrix(0, ncol = ntvec, nrow = NMCMC)
chCIHR <- matrix(0, ncol = ntvec, nrow = NMCMC)
SCIHR <- matrix(0, ncol = ntvec, nrow = NMCMC)
qCIHR <- matrix(0, ncol = ntvec, nrow = NMCMC)

for(k in 1:NMCMC){
  paramsHRj  <- c(lambda = postHT[[j]][k,1], kappa = postHT[[j]][k,2], alpha = postHT[[j]][k,3],
                  beta = postHT[[j]][k,4])
  initHRj      <- c(h = h0, q = q0, H = 0 )
  outj <- ode(initHRj, tvec, hazmodHR, paramsHRj, method = "lsode", jacfunc = jacODE, jactype = "fullusr")
  
  hCIHR[k, ] <- outj[,2]
  qCIHR[k, ] <- outj[,3]
  chCIHR[k, ] <- outj[,4]
  SCIHR[k, ] <- exp(-outj[,4])
  
} 


numHR <- hCIHR*exp(-chCIHR)
denHR <- exp(-chCIHR)

hpredHR <- colMeans(numHR)/colMeans(denHR)

#--------------------------------------------------------------------------------------
# Distances
#--------------------------------------------------------------------------------------


bsh <- Vectorize(function(t) splinefun(x = fit$time, y = fit$hazard)(t))
hts <- Vectorize(function(t) splinefun(tvec, hT)(t))
hps <- Vectorize(function(t) splinefun(tvec, hpredHR)(t))

difbs <- Vectorize(function(t) abs(bsh(t) - hts(t)))
difp <- Vectorize(function(t) abs(hps(t) - hts(t)))


distbs[j] <- integrate(difbs, 0.001, cens)$value
distp[j] <- integrate(difp, 0.001, cens)$value

}

#setwd("~/Dropbox/ODESurv/Codes/Simulations/HT/cens50/n250")
setwd("~/Dropbox/ODESurv/ODESurv/Codes/Simulations/HT/cens25/n250")
save.image("HTn250c25_comparison.RData")
