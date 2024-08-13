#=
****************************************************************************
Required packages
****************************************************************************
=#

# using Pkg
# Pkg.add("CSV")
# Pkg.add("LSODA")
# Pkg.add("Optim")
# Pkg.add("Distributions")
# Pkg.add("Random")
# Pkg.add("AdaptiveMCMC")
# Pkg.add("Tables")
# Pkg.add("DelimitedFiles")
# Pkg.add("Statistics")
# Pkg.add("Survival")
# Pkg.add("DataFrames")
# Pkg.add("FreqTables")
# Pkg.add("Sundials")
# Pkg.add("ForwardDiff")

using Plots
using DifferentialEquations
using LinearAlgebra
using CSV
using LSODA
using Optim
using Distributions
#using JTwalk
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
****************************************************************************
Hazard-Response model
****************************************************************************
=#

# Hazard-Response ODE model
function HazResp(dh, h, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dh[1] = lambda * h[1] * (1 - h[1] / kappa) - alpha * h[1] * h[2] # hazard
    dh[2] = beta * h[2] * (1 - h[2] / kappa) - alpha * h[1] * h[2] # response
    dh[3] = h[1] # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHR(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    h = u[1]
    q = u[2]

    # Jacobian
    J[1, 1] = lambda * (1 - 2 * h / kappa) - alpha * q
    J[1, 2] = -alpha * h
    J[1, 3] = 0.0
    J[2, 1] = -alpha * q
    J[2, 2] = beta * (1 - 2 * q / kappa) - alpha * h
    J[2, 3] = 0.0
    J[3, 1] = 1.0
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJ = ODEFunction(HazResp; jac=jacHR)

# Initial conditions (h,q,H)
u0 = [1.0e-2, 1.0e-6, 0.0]



#=
****************************************************************************
Hazard-Response model (log h and log q)
****************************************************************************
=#

# Hazard-Response ODE model
function HazRespL(dlh, lh, p, t)
    # Model parameters
    lambda, kappa, alpha, beta = p

    # ODE System
    dlh[1] = lambda * (1 - exp(lh[1]) / kappa) - alpha * exp(lh[2]) # log hazard
    dlh[2] = beta * (1 - exp(lh[2]) / kappa) - alpha * exp(lh[1]) # log response
    dlh[3] = exp(lh[1]) # cumulative hazard
    return nothing
end

# Jacobian for Hazard-Response model

function jacHRL(J, u, p, t)
    # Parameters
    lambda, kappa, alpha, beta = p
    # state variables
    lh = u[1]
    lq = u[2]

    # Jacobian
    J[1, 1] = -lambda * exp(lh) / kappa
    J[1, 2] = -alpha * exp(lq)
    J[1, 3] = 0.0
    J[2, 1] = -alpha * exp(lh)
    J[2, 2] = -beta * exp(lq) / kappa
    J[2, 3] = 0.0
    J[3, 1] = exp(lh)
    J[3, 2] = 0.0
    J[3, 3] = 0.0
    nothing
end

# Hazard-Response model with explicit Jacobian
HRJL = ODEFunction(HazRespL; jac=jacHRL)

# Initial conditions (h,q,H)
lu0 = [log(1.0e-2), log(1.0e-6), 0.0]



#=
****************************************************************************
Negative log-likelihood functions
****************************************************************************
=#

# Negative log likelihood function
mlog_lik = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)

        sol = solve(ODEProblem(HRJ, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #     sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]), Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(log.(OUT[1, status]))

        ll_chaz = sum(OUT[3, :])

        mloglik = -ll_haz + ll_chaz
    end
    return mloglik
end


# Negative log likelihood function (log h and log q)
mlog_likL = function (par::Vector{Float64})
    if any(par .> 5.0)
        mloglik = Inf64
    else
        # Parameters for the ODE
        odeparams = exp.(par)


        sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
        #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
        OUT = sol(df.time)


        # Terms in the log log likelihood function
        ll_haz = sum(OUT[1, status])

        ll_chaz = sum(OUT[3, :])


        mloglik = -ll_haz + ll_chaz
    end
    return mloglik
end


#  log likelihood function (log h and log q)
log_likL = function (par)
    # Parameters for the ODE
    odeparams = exp.(par)


    sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
    #  sol = solve(ODEProblem(HRJL, lu0, tspan0[i, :], odeparams[i, :]),Tsit5())
    OUT = sol(df.time)


    # Terms in the log log likelihood function
    ll_haz = sum(OUT[1, status])

    ll_chaz = sum(OUT[3, :])


    ll = ll_haz - ll_chaz

return ll
end

#= 
**********************************************************************************
Some numerical analysis
**********************************************************************************
=#


#initmle = zeros(7)

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
using Turing, MCMCChains, StatsPlots, Distributions

distprior = Gamma(2,2)

# # define the log-posterior function as a Turing model:
# @model function bayesian_model(times, status)
#     # prior (defined on the positive parameters)
#     # odeparams = exp.(par)
#    odeparams ~ filldist(distprior, 4)

#     # if any(par .> 3.0)
#     #     Turing.@addlogprob! -Inf #Skip this sample
#     # else
        
#         prob = ODEProblem(HazRespL, lu0, (0.0, tmax), odeparams)
#         sol = solve(prob; alg_hints=[:stiff])
#         OUT = sol(times)

#         ll_haz = sum(OUT[1, status .== 1])
#         ll_chaz = sum(OUT[3, :])
#         # l_prior = sum(logpdf.(distprior, odeparams))
#         # l_JAC = sum(par)

#         # for loop?
        
# #        lp = ll_haz - ll_chaz + l_prior
#         lp = ll_haz - ll_chaz 
#         Turing.@addlogprob! lp
#     end

# define the log-posterior function as a Turing model:
@model function bayesian_model(log_likL)
    # prior (defined on the positive parameters)
   odeparams ~ filldist(distprior, 4)
        params = log.(odeparams)

        Turing.@addlogprob!(log_likL(params))
    end

# Run the MCMC sampler
Random.seed!(123)

#model = bayesian_model(times, status)
model = bayesian_model(log_likL)
NMC = 110000
burn = 10000
thin = 100

chain = sample(model, NUTS(), NMC; init_params=MLE)
#burned_chain = Chains(chain[burn+1:end; thin=thin]) #returns an error

# Optionally, plot the results
#plot(burned_chain)


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

# To be discussed
using MCMCDiagnosticTools