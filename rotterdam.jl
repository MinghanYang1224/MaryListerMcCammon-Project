#=
****************************************************************************
Required packages
****************************************************************************
=#

using Plots
using DifferentialEquations
using LinearAlgebra
using CSV
using LSODA
using Optim
using Distributions
using Random
using AdaptiveMCMC
using Tables
using Statistics
using Survival
using DataFrames
using ForwardDiff
using StatsPlots
using Turing
using MCMCChains

include("routinesODESurv.jl")

#=
****************************************************************************
Data preparation
****************************************************************************
=#

#= Data =#
df_full = CSV.File("rotterdamFull.csv");


df = DataFrame(time=collect(df_full.dtime) ./ 365.25,
    status=collect(df_full.death))

# Sorting df by time
sorted_indices = sortperm(df[:, :time]) # increasing order

df = df[sorted_indices, :]

# Sample size
n = size(df)[1]

#= Vital status =#
status = collect(Bool, (df.status)); #0，1，1，0，。。。

#= Survival times =#
times = df.time; 

# Time grid
tspan0 = vcat(0.0, df.time); #Concatenate arrays or numbers vertically 链接成一个长的向量
tmax = maximum(df.time)

#= 
**********************************************************************************
Some numerical analysis
**********************************************************************************
=#


initmle = [0.5381167, -2.3121004, 1.9444865, 1.5792623] #logarithm of the mle's

optimiser = optimize(mlog_lik, initmle * 0.0, method=NelderMead(), iterations=10000)

logMLE = optimiser.minimizer # the optimal parameter values for log(lambda, kappa, alpha, beta)
MLE = exp.(logMLE)

# Save MLE
writedlm("logMLE.txt", optimiser.minimizer)
writedlm("MLE.txt", MLE)



#=
**********************************************************************************
Turing for posterior sampling
**********************************************************************************
=#

# Run the MCMC sampler
Random.seed!(123)

# define the model for turing
model = bayesian_model(log_likL)

# Total number of iterations:
NMC = 110000
# Number of burned samples
burn = 10000
# Thinning
thin = 100

# sampling using the NUTS() method. 
chain = sample(model, NUTS(), NMC; init_params=MLE)
run(`afplay /System/Library/Sounds/Glass.aiff`)
# Overview of the chain
summarystats(chain)

# Plot the histograms
h1a = histogram(chain.value[ burn:thin:end,1])
h2a = histogram(chain.value[ burn:thin:end,2])
h3a = histogram(chain.value[ burn:thin:end,3])
h4a = histogram(chain.value[ burn:thin:end,4])

plot(h1a, h2a, h3a, h4a, layout=(2, 2), legend=false)


# Plot the smoothed densities
tp1a = plot(chain.value[ burn:thin:end,1])
tp2a = plot(chain.value[ burn:thin:end,2])
tp3a = plot(chain.value[ burn:thin:end,3])
tp4a = plot(chain.value[ burn:thin:end,4])

plot(tp1a, tp2a, tp3a, tp4a, layout=(2, 2), legend=false)

# compare with the MLE 
hcat(MLE,vec(mean(chain.value[ burn:thin:end,1:4],dims=1)))

# Save posterior samples
postsamp = Tables.table(transpose(chain.value[ burn:thin:end,1:4]))

CSV.write("postsamp.csv", postsamp)

#=
**********************************************************************************
Plot posteior vs prior
**********************************************************************************
=#

# read posterior samples as dataframe.
postsamples = CSV.read("postsamp.csv", DataFrame)

param_names = ["lambda", "kappa", "alpha", "beta"]
priors = [Gamma(2, 2) for _ in 1:4]

# define ranges in x axis for the four plots:
x_ranges = [
    range(0, stop=6,   length=1000),  # for lambda
    range(0, stop=0.5, length=1000),  # for kappa
    range(0, stop=11,  length=1000),  # for alpha
    range(0, stop=10,  length=1000)   # for beta
]

# Plot posterior vs prior
plot_list = [plot_posterior_with_prior(postsamples, priors, i, param_names[i], x_ranges[i]) for i in 1:4]
plot(plot_list..., layout=(2, 2), size = (1000,800))

#=
**********************************************************************************
Predictive Hazard Function
**********************************************************************************
=#

# Number of posterior samples
M = ncol(postsamples)

# define time vector
t_vector = [0.01:0.1:20.5;]
nt = length(t_vector)

# Posterior predictive hazard
Predictive_hazard = zeros(nt)
Predictive_hazardU = zeros(nt)
Predictive_hazardL = zeros(nt)
for i in 1:nt
    Predictive_hazard[i] = PredHR(t_vector[i])[1]
    Predictive_hazardU[i] = PredHR(t_vector[i])[2]
    Predictive_hazardL[i] = PredHR(t_vector[i])[3]
end

# Postrior predictive response
Predictive_response = zeros(nt)
for i in 1:nt
    Predictive_response[i] = PredResp(t_vector[i])
end

# Posterior predictive survival 
Pred_S = zeros(nt)
Pred_SU = zeros(nt)
Pred_SL = zeros(nt)
for i in 1:nt
    Pred_S[i] = PredSurv(t_vector[i])[1]
    Pred_SU[i] = PredSurv(t_vector[i])[2]
    Pred_SL[i] = PredSurv(t_vector[i])[3]
end

# Plot the posterior predictive hazard
plot(t_vector, Predictive_hazard, label="Predictive Hazard", lw=2, ylim=(0,0.09))
xlabel!("time")
ylabel!("Predictive Hazard Function")
title!("Posterior Predictive Hazard")

# Plot the posterior predictive hazard and response
plot(t_vector, Predictive_hazard, label="Predictive Hazard", lw=2, ylim=(0,0.13))
plot!(t_vector, Predictive_response, label="Predictive Response", lw=2)
xlabel!("time")
ylabel!("Predictive Functions")
title!("Posterior Predictive Functions")

# Plot the posterior predictive survival
plot(t_vector, Pred_S, label="Predictive Survival", lw=2, ylim=(0,1))
xlabel!("time")
ylabel!("Predictive Survival Function")
title!("Posterior Predictive Survival")



# Compare with Kaplan-Meier estimator
using Survival
km = fit(KaplanMeier, df.time, df.status)
ktimes = sort(unique(times))
ksurvival_probs = km.survival

# Comparison plot
plot(ktimes, ksurvival_probs,
    xlabel = "Time (years)", ylabel = "Survival", title = "Kaplan-Meier VS Posterior Predictive Survival",
    label = "Kaplan-Meier",
  xlims = (0.0001,maximum(times)),   xticks = 0:2:maximum(times), linewidth=3,
  linecolor = "gray", ylims = (0,1), linestyle=:solid)
plot!(t_vector, Pred_S, label="Predictive Survival", lw=2)



# Add credible intervals to predictive hazard and predictive survival
plot(t_vector, Predictive_hazard, label="Predictive Hazard", lw=2, ylim=(0,0.09))
plot!(t_vector, Predictive_hazardU, fillrange=Predictive_hazardL, fillalpha=0.3, label="95% CI", color=:grey, linecolor=:transparent)
xlabel!("time")
ylabel!("Predictive Hazard Function")
title!("Posterior Predictive Hazard")

plot(t_vector, Pred_S, label="Predictive Survival", lw=2, ylim=(0,1))
plot!(t_vector, Pred_SU, fillrange=Pred_SL, fillalpha=0.3, label="95% CI", color=:grey, linecolor=:transparent)
xlabel!("time")
ylabel!("Predictive Survival Function")
title!("Posterior Predictive Survival")


# Set threads to 8 in terminal using "julia --threads 8"
begin
    chain_parallel = sample(model, NUTS(), MCMCThreads(), 110000, 2; init_params=MLE) # four chains in parallel
    summarystats(chain_parallel)
end

# time used: 0:08:10

chain_parallel