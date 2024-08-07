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
****************************************************************************
Log-posterior functions -- TO BE CORRECTED and COMPARED
****************************************************************************
=#


# # Priors
# distprior = Gamma(2,2)

# # Log-posterior
# log_post = function (par::Vector{Float64})
#     if any(par .> 3.0)
#         lp = -Inf64
#     else
#         # Parameters for the ODE
#         odeparams = exp.(par)

#         sol = solve(ODEProblem(HRJ, u0, [0.0, tmax], odeparams); alg_hints=[:stiff])
#         #     sol = solve(ODEProblem(HRJ, u0, tspan0[i, :], odeparams[i, :]), Tsit5())
#         OUT = sol(df.time)


#         # Terms in the log log likelihood function
#         ll_haz = sum(log.(OUT[1, status]))

#         ll_chaz = sum(OUT[3, :])

#         # log prior
#         l_prior = sum(logpdf.(distprior, odeparams))

#         # log-Jacobian
#         l_JAC = sum(par)

#         lp = ll_haz - ll_chaz + l_prior + l_JAC
#     end
#     return lp
# end


# # Log-posterior (log h and log q)
# log_postL = function (par::Vector{Float64})
#     if any(par .> 3.0)
#         lp = -Inf64
#     else
#         # Parameters for the ODE
#         odeparams = exp.(par)

#         sol = solve(ODEProblem(HRJL, lu0, [0.0, tmax], odeparams); alg_hints=[:stiff])
#         OUT = sol(df.time)

#         # Terms in the log log likelihood function
#         ll_haz = sum(OUT[1, status])

#         ll_chaz = sum(OUT[3,:])

#         # log prior
#         l_prior = sum(logpdf.(distprior, odeparams))

#         # log-Jacobian
#         l_JAC = sum(par)

#         lp = ll_haz - ll_chaz + l_prior + l_JAC
#     end
#     return lp
# end

# MLE = vec(readdlm("MLE.txt"))
# log_post(MLE) #-4785.3241550528965
# log_postL(MLE) #-4785.308725489181

# # Run NMC iterations of the Adaptive Metropolis:

# NMC = 75000
# Random.seed!(123)
# out = adaptive_rwm(MLE, log_post, NMC; algorithm=:ram)

# # Run NMC iterations of the Adaptive Metropolis:
# #init0 = MLER

# #NMC = 75000
# #Random.seed!(1234)
# #out = adaptive_rwm(initmle, log_postL, NMC; algorithm=:am)

# # Calculate '95% credible intervals':

# mapslices(x -> "$(mean(x)) ± $(1.96std(x))", out.X, dims=2)


# hcat(MLE,vec(mean(out.X,dims=2)))

# burn = 1
# thin = 50

# h1a = histogram(exp.(out.X[1, burn:thin:end]))
# h2a = histogram(exp.(out.X[2, burn:thin:end]))
# h3a = histogram(exp.(out.X[3, burn:thin:end]))
# h4a = histogram(exp.(out.X[4, burn:thin:end]))


# plot(h1a, h2a, h3a, h4a, layout=(3, 3), legend=false)


# tp1a = plot(out.X[1, burn:thin:end])
# tp2a = plot(out.X[2, burn:thin:end])
# tp3a = plot(out.X[3, burn:thin:end])
# tp4a = plot(out.X[4, burn:thin:end])


# plot(tp1a, tp2a, tp3a, tp4a, layout=(3, 3), legend=false)

# # Save posterior samples
# postsamp = Tables.table(transpose(out.X[:, burn:thin:end]))

# CSV.write("postsamp.csv", postsamp)




# #= 
# **********************************************************************************
# MLE analysis
# **********************************************************************************
# =#

# #= 
# **********************************************************************************
# Posterior analysis
# **********************************************************************************
# =#

# #= Data =#
# postsamp = CSV.File("postsamp.csv");

# # Histograms
# h1a = histogram(postsamp.Column1)
# h2a = histogram(postsamp.Column2)
# h3a = histogram(postsamp.Column3)
# h4a = histogram(postsamp.Column4)



# plot(h1a, h2a, h3a, h4a, layout=(3, 3), legend=false)


#=
**********************************************************************************
Turing for posterior sampling
**********************************************************************************
=#
using Turing, MCMCChains, StatsPlots, Distributions

distprior = Gamma(2,2)

# define the log-posterior function as a Turing model:
@model function bayesian_model(times, status)

   odeparams ~ filldist(distprior, 4)
        
        prob = ODEProblem(HazRespL, lu0, (0.0, tmax), odeparams)
        sol = solve(prob; alg_hints=[:stiff])
        OUT = sol(times)

        ll_haz = sum(OUT[1, status .== 1])
        ll_chaz = sum(OUT[3, :])

        lp = ll_haz - ll_chaz 
        Turing.@addlogprob! lp
    end


# Run the MCMC sampler
Random.seed!(123)

model = bayesian_model(times, status)
NMC = 110000
burn = 10000
thin = 100

chain = sample(model, NUTS(), NMC; init_params=MLE)
#burned_chain = Chains(chain[burn+1:end; thin=thin]) #returns an error


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

# Plot the marginal posterior distributions of the parameters. 
# Function to plot posterior and prior for a parameter
using StatsPlots, DataFrames
postsamples = CSV.read("postsamp.csv", DataFrame)

function plot_posterior_with_prior(postsamples, priors, param_idx, param_name, x_range)
    posterior_samples = postsamples[param_idx,:]
    prior = priors[param_idx]
    
    prior_pdf = pdf.(prior, x_range)

    # Plot
    density(Vector(posterior_samples), normalize=true, label="Posterior", legend=:topright)
    #histogram(posterior_samples, normalize=true, bins=30, label="Posterior", alpha=0.6)
    plot!(x_range, prior_pdf, label="Prior", lw=2)
    xlabel!(param_name)
    ylabel!("Density")
    title!("Posterior and Prior for $param_name")
end

# Plotting each parameter
param_names = ["lambda", "kappa", "alpha", "beta"]
priors = [Gamma(2, 2) for _ in 1:4]
# define ranges in x axis for the four plots:
x_ranges = [
    range(0, stop=6, length=1000),  # for lambda
    range(0, stop=0.5, length=1000),  # for kappa
    range(0, stop=11, length=1000),  # for alpha
    range(0, stop=10, length=1000)   # for beta
]
# Plot
plot_list = [plot_posterior_with_prior(postsamples, priors, i, param_names[i], x_ranges[i]) for i in 1:4]
# Display the plots
plot(plot_list..., layout=(2, 2), size = (1000,800))

# To be discussed
using MCMCDiagnosticTools

#=
**********************************************************************************
Predictive Hazard Function
**********************************************************************************
=#

# Number of posterior samples
M = ncol(postsamples)

# Predictive hazard:
function PredHR(t)
    OUT = zeros(Float64, M, 3) # M times 3 matrix of zeros.
    for i in 1:M
        sol = solve(ODEProblem(HRJ, u0, t, postsamples[:,i]), Tsit5())
        OUT[i,:] = reduce(vcat, sol.u[end,:]) # take the values of the three functions at the last time point t
    end

    hPred = mean(exp.(-OUT[:,3]) .* OUT[:,1]) / mean(exp.(-OUT[:,3]))
    return hPred 
    # returns the predictive hazard function evaluated at time t
end

# Predictive survival:
function PredSurv(t)
    OUT = zeros(Float64, M, 3) # M times 3 matrix of zeros.
    for i in 1:M
        sol = solve(ODEProblem(HRJ, u0, t, postsamples[:,i]), Tsit5())
        OUT[i,:] = reduce(vcat, sol.u[end,:]) # take the values of the three functions at the last time point t
    end

    SPred = mean(exp.(-OUT[:,3]))
    return SPred 
    # returns the predictive hazard function evaluated at time t
end

t_vector = [0.01:0.1:20.5;]
nt = length(t_vector)

Predictive_hazard = zeros(nt)
for i in 1:nt
    Predictive_hazard[i] = PredHR(t_vector[i])
end

Post_S = zeros(nt)
for i in 1:nt
    Post_S[i] = PredSurv(t_vector[i])
end

# Plot the posterior predictive hazard
plot(t_vector, Predictive_hazard, label="Predictive Hazard", lw=2)
xlabel!("time")
ylabel!("Predictive Hazard Function")
title!("Posterior Predictive Hazard")

# Plot the posterior predictive survival
plot(t_vector, Post_S, label="Predictive Survival", lw=2)
xlabel!("time")
ylabel!("Predictive Survival Function")
title!("Posterior Predictive Survival")
