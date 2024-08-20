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

# define the log-posterior function as a Turing model:
distprior = Gamma(2,2)

@model function bayesian_model(log_likL)
    # prior (defined on the positive parameters)
   odeparams ~ filldist(distprior, 4)
        params = log.(odeparams)

        Turing.@addlogprob!(log_likL(params))
    end

    
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