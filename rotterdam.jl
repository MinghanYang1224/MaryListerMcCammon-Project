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
using DelimitedFiles
using Statistics
using Survival
using DataFrames
using FreqTables
using Sundials
using ForwardDiff

using Turing, MCMCChains, StatsPlots, MCMCDiagnosticTools

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


initmle = [0.5381167, -2.3121004, 1.9444865, 1.5792623]

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

#model = bayesian_model(times, status)
model = bayesian_model(log_likL)
NMC = 110000
burn = 10000
thin = 100

chain = sample(model, NUTS(), NMC; init_params=MLE)


# Summaries

h1a = histogram(chain.value[ burn:thin:end,1])
h2a = histogram(chain.value[ burn:thin:end,2])
h3a = histogram(chain.value[ burn:thin:end,3])
h4a = histogram(chain.value[ burn:thin:end,4])



plot(h1a, h2a, h3a, h4a, layout=(2, 2), legend=false)

tp1a = plot(chain.value[ burn:thin:end,1])
tp2a = plot(chain.value[ burn:thin:end,2])
tp3a = plot(chain.value[ burn:thin:end,3])
tp4a = plot(chain.value[ burn:thin:end,4])


plot(tp1a, tp2a, tp3a, tp4a, layout=(2, 2), legend=false)

hcat(MLE,vec(mean(chain.value[ burn:thin:end,1:4],dims=1)))

# Save posterior samples
postsamp = Tables.table(transpose(chain.value[ burn:thin:end,1:4]))

CSV.write("postsamp.csv", postsamp)
