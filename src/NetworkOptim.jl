
module NetworkOptim

using LinearAlgebra, Statistics
using ClassicalAndQuantumFIMs, Optim, GalacticOptim, BlackBoxOptim, JLD2, Serialization, Logging, Dates, ThreadsX
using Flux: ADAM
using ..BiasedNetworkModels: BiasedSheetModel, BiasedPrismModel, BiasedRingModel
include("utilities.jl")

const default_run_params = (
    #Essentials
    SaveName = "test",
    ChainLength = 12,
    NumChains = 2,
    dE_ratio = 1.0,
    ChainSep = 1.0,
    CouplingFunc = distance_only_coupling,
    Geom = :sheet,

    #Optimization params
    WithEs = true,
    WithSep = false,
    WithDipoles = false,
    EnergyVarSites = nothing, #What's the best way to set these?
    DipoleVarSites = nothing,
    SingleObjTimeLimit = 60,
    SingleObjMaxIters = 10^7, #Prefer to rely on time limit
    MaxSteps = 10^5,
    NumEnsemble = 15,
    Symm = true,
    SearchRange = nothing,
)

function create_run_params(; kwargs...) #kwargs are used to overwrite defaults

    RP = (; default_run_params..., kwargs...) #Create named tuple 

    #Check we haven't made any typos in kwargs
    for k in keys(kwargs)
        !(k in keys(default_run_params)) && error("Key '$(k)' not found in default run params.")
    end

    #Sanity checks
    (RP.WithEs && RP.EnergyVarSites === nothing) && error("Need to provide EnergyVarSites or set WithEs = false")
    (!RP.WithEs && RP.EnergyVarSites !== nothing) && error("Need to set WithEs = true when providing EnergyVarSites")
    (RP.WithSep && RP.NumChains < 2) && error("Optimizing chain separation makes no sense for a single chain...")
    (RP.WithDipoles && RP.DipoleVarSites === nothing) && error("Need to provide DipoleVarSites or set WithDipoles = false")
    (!RP.WithDipoles && RP.DipoleVarSites !== nothing) && error("Need to set WithDipoles = true when providing DipoleVarSites")

    any(isfile.(RP.SaveName .* ["-res.sjl", "-res-bak.txt", "-trace.txt", "-res.jld2"])) && error("File name already exists - move existing file or choose different save name.")

    return RP
end



# -------------------------------------------------------------------------------------------------------------------- #
#                                        Opimization-specific utility functions                                        #
# -------------------------------------------------------------------------------------------------------------------- #



function init_model(run_params)

    if run_params.Geom == :sheet
        return BiasedSheetModel(run_params.ChainLength, run_params.NumChains, run_params.dE_ratio; interchain_coupling=1/run_params.ChainSep^3, coupling_func=run_params.CouplingFunc)
    elseif run_params.Geom == :prism
        return BiasedPrismModel(run_params.ChainLength, run_params.NumChains, run_params.dE_ratio; interchain_coupling=1/run_params.ChainSep^3, coupling_func=run_params.CouplingFunc)
    elseif occursin(string(run_params.Geom), "ring")
        return BiasedRingModel(run_params.ChainLength, run_params.dE_ratio, run_params.Geom; coupling_func=run_params.CouplingFunc)
    end
end


function get_x(model, run_params)

    x = Float64[]
    if run_params.WithEs 
        Es = site_energies(model)
        idxs = run_params.EnergyVarSites
        run_params.Symm && (idxs = idxs[1:length(idxs)÷model.Ham.num_chains])
        append!(x, Es[idxs])
    end
    run_params.WithSep && push!(x, get_interchain_coupling(model))
    run_params.WithDipoles && append!(x, reduce(vcat, [get_dipole_angles(model, site) for site in run_params.DipoleVarSites]))

    return x
end


function update_x!(model, x, run_params)

    #Sanity check
    expected_length = 0
    if run_params.WithEs 
        if run_params.Symm
            expected_length += length(run_params.EnergyVarSites) ÷ model.Ham.num_chains
        else
            expected_length += length(run_params.EnergyVarSites)
        end
    end
    expected_length += run_params.WithSep 
    run_params.WithDipoles && (expected_length += 2*length(run_params.DipoleVarSites))
    length(x) != expected_length && error("Length of x vector is incorrect.")


    i_start = 1 #For tracking different sections of x vector

    if run_params.WithEs
        L = length(run_params.EnergyVarSites)
        run_params.Symm && (L ÷= model.Ham.num_chains)
        new_Es = x[i_start:L]
        run_params.Symm && (new_Es = repeat(new_Es, outer=model.Ham.num_chains)) #Remember to repeat energies if we want symmetry
        vary_site_energies!(model, new_Es, run_params.EnergyVarSites, update_H=false, update_L=false)
        i_start += L
    end

    if run_params.WithSep
        new_sep = x[i_start]
        vary_interchain_coupling!(model, 1/new_sep^3, update_H=false, update_L=false)
        i_start += 1
    end

    if run_params.WithDipoles
        new_dipole_angles = x[i_start:end]
        new_θs = new_dipole_angles[1:2:end]
        new_ϕs = new_dipole_angles[2:2:end]
        for (site, θ, ϕ) in zip(run_params.DipoleVarSites, new_θs, new_ϕs)
            vary_dipole_orientation!(model, site, CartesianFromSpherical()(Spherical(1, θ, ϕ))..., update_H=false, update_L=false)
        end
    end

    #Update model fields
    model.Ham = update_H!(model.Ham)
    model.L = transport_generator(model)

    return model
end
#Non-mutating version
update_x(model, x, run_params) = update_x!(copy(model), x, run_params)



# Randomly peturbs all site energies with s std dev of `dis` % of average excited state energy (i.e. with E0 subtracted)
function perturb_Es!(model; dis=1e-2, kwargs...)
    Es = site_energies(model)
    E0 = minimum(Es)
    N = numsites(model)
    E_avg = mean(Es) - E0
    return vary_site_energies!(model, Es .+ dis*E_avg*randn(N), 1:N; kwargs...)
end
#Non-mutating version
perturb_Es(model; kwargs...) = perturb_Es!(copy(model); kwargs...)


function perturb_chain_sep!(model; dis=1e-2, kwargs...)
    current_sep = 1/cbrt(get_interchain_coupling(model))
    new_sep = current_sep*(1+dis*randn())
    return vary_interchain_coupling!(model, 1/(new_sep)^3; kwargs...)
end
#Non-mutating version
perturb_chain_sep(model; kwargs...) = perturb_chain_sep!(copy(model); kwargs...)


#Randomly perturbs all dipole orientations with std dev of `dis` / 2π radians
function perturb_dipole_angles!(model; dis=1e-2, kwargs...)

    all_angles = reduce(vcat, [get_dipole_angles(model, s) for s in 1:numsites(model)])
    θs = all_angles[1:2:end]
    ϕs = all_angles[2:2:end]

    for (site, (θ, ϕ)) in enumerate(zip(θs, ϕs))
        vary_dipole_θ!(model, site, θ + 2*π*dis*randn(); kwargs...)
        vary_dipole_ϕ!(model, site, ϕ + 2*π*dis*randn(); kwargs...)
    end

end
#Non-mutating version
perturb_dipole_angles(model; kwargs...) = perturb_dipole_angles!(copy(model); kwargs...)



function perturb_x!(model, run_params; dis=1e-2)

    # x0 = get_x(model, run_params)
    run_params.WithEs && perturb_Es!(model, dis=dis, update_H=false, update_L=false)
    run_params.WithSep && perturb_chain_sep!(model, dis=dis, update_H=false, update_L=false)
    run_params.WithDipoles && perturb_dipole_angles!(model, dis=dis, update_H=false, update_L=false)

    model.Ham = update_H!(model.Ham)
    model.L = transport_generator(model)

    return model
end
#Non-mutating version
perturb_x(model, RP; kwargs...) = perturb_x!(copy(model), RP; kwargs...)





function FIM_params(model, run_params)
    FIM_params = DiffParam[]
    run_params.WithEs && append!(FIM_params, SiteEnergyDeriv(model, 1))
    run_params.WithSep && push!(FIM_params, InterchainCouplingDeriv(model, 1))
    run_params.WithDipoles && append!(FIM_params, DipoleAngleDeriv(model, 1))
    return FIM_params
end


# -------------------------------------------------------------------------------------------------------------------- #
#                                           Objective functions and callbacks                                          #
# -------------------------------------------------------------------------------------------------------------------- #


function current_and_QFIM_trace(model::OQSmodel, run_params::NamedTuple)#::Tuple{Float64, Float64}
    Iss = ss_current(model)
    QFIM = QuantumFIM(model, steady_state, FIM_params(model, run_params))::Matrix{ComplexF64} #Annotate return type
    return (-Iss, real(tr(QFIM))) #Use -ve Iss since we want to maximize this value
end

#Version which sets new x before calc of objective values
current_and_QFIM_trace(m::OQSmodel, x::Vector{R} where R <: Real, RP::NamedTuple) = current_and_QFIM_trace(update_x(m, x, RP), RP)


# Flux algs don't allow any kwargs apart from maxiters, abstol & reltol, so need to implement trace saving and time limit with a callback function
function early_stopping_cb(params, obj, time_limit, start_time, obj_hist)

    push!(obj_hist, obj)

    L = length(obj_hist)
    run_time = datetime2unix(now()) - start_time

    #Check stopping criteria
    if run_time > time_limit
        str = "Time limit ($(time_limit) s) reached --> quitting optimization run.\n"
        str *= "Final objective value = $(obj)\n"
        str *= "Final x vector: $(round.(params, sigdigits=4))\n"
        println(str)
        return true #Returning true causes optimization to exit

    elseif L > 1000 && all(minimum(obj_hist) .< obj_hist[L-1000:L])
        str = "No improvement in objective value in past 1000 iterations --> quitting optimization run.\n"
        str *= "Final objective value = $(obj)\n"
        str *= "Final x vector: $(round.(params, sigdigits=4))\n"
        println(str)
        return true #Returning true causes optimization to exit
    end

    return false 
end




# -------------------------------------------------------------------------------------------------------------------- #
#                                             Main optimization procedures                                             #
# -------------------------------------------------------------------------------------------------------------------- #


function run_ensemble_opt(model, run_params; alg=ADAM(0.001), dis=1e-2)

    opt_func = OptimizationFunction((x, p) -> -1*ss_current(update_x!(p, x, run_params)), GalacticOptim.AutoFiniteDiff())

    #Optimize unperturbed model
    tmp_model = copy(model) #Avoid mutating input model
    x0 = get_x(tmp_model, run_params)
    prob = GalacticOptim.OptimizationProblem(opt_func, x0, tmp_model)
    obj_hist_unperturbed = [] #Create list for storing history
    start_time = Dates.datetime2unix(Dates.now())
    opt_task = Threads.@spawn solve(prob, alg, maxiters=run_params.SingleObjMaxIters, cb = (p, f_x) -> early_stopping_cb(p, f_x, run_params.SingleObjTimeLimit, start_time, obj_hist_unperturbed))

    #Optimize from perturbed starting points
    sol_list = ThreadsX.map(1:run_params.NumEnsemble) do run
        tmp_model = perturb_x(model, run_params, dis=dis)
        x0 = get_x(tmp_model, run_params)
        prob = GalacticOptim.OptimizationProblem(opt_func, x0, tmp_model)
        obj_hist = [] #Create local list for storing history
        start_time = Dates.datetime2unix(Dates.now())
        solve(prob, alg, maxiters=run_params.SingleObjMaxIters, cb = (p, f_x) -> early_stopping_cb(p, f_x, run_params.SingleObjTimeLimit, start_time, obj_hist))
    end

    #Wait for unperturbed opt to finish too, then add it to sol_list
    push!(sol_list, fetch(opt_task))

    return sol_list

end


function run_multi_obj(RP::NamedTuple; obj_func=current_and_QFIM_trace, kwargs...) #kwargs are passed to run_ensemble_opt

    start_model = init_model(RP)


    # #Set search range for multi-obj opt automatically if it's not set
    # if RP.SearchRange === nothing
    #     SR = []
    #     if RP.WithEs 
    #         Es = site_energies(start_model)[RP.EnergyVarSites]
    #         RP.Symm && (Es = Es[1:length(RP.EnergyVarSites)÷start_model.Ham.num_chains])
    #         append!(SR, )
    #     end
    # end

    open("$(RP.SaveName)-trace.txt", "a") do f

        #Send all printed output to a text file
        redirect_stdout(f)
        #Send all warnings to the same text file
        logger = SimpleLogger(f)
        global_logger(logger)

        #Needs to be defined within 'open' block so that f is accesible
        function PF_callback(oc::BlackBoxOptim.OptController)
            println("Range of enhancement factors found so far: ", extrema(first.(getfield.(fitness.(pareto_frontier(oc.evaluator.archive)), :orig))), "\n")
            flush(f) #Force printing to file regularly for progress monitoring
        end
    
        # Perform single-obj ensemble opt
        sol_list = run_ensemble_opt(start_model, RP; kwargs...)

        # Set up multi-obj opt run
        opt_ctrl = bbsetup(
            x -> obj_func(start_model, x, RP), #Function to be optimized
            Method=:borg_moea,
            FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true), #, aggregator=f -> f[1]), #Use aggregator to display largest enhancement factor in trace (rather than equally weighted sum of both objectives)
            SearchRange=SEARCH_RANGE,
            # PopulationSize=100, #Default is 50
            # MaxSteps = 2*10^5, #Is ignored in favour of MaxTime by BlackBoxOptim (unless MaxTime=0)
            # MaxStepsWithoutProgress=10^5, #default is 10^4
            # MaxNumStepsWithoutFuncEvals = 5000, #Default is 100
            # ϵ=[1e-3, 1e-3], #Size of 'epsilon box' for measuring fitness progress
            ϵ=1e-4, #Size of 'epsilon box' for measuring fitness progress
            MaxTime=RP.t2,
            TraceInterval=RP.t2/50,
            # CallbackFunction = opt_ctrl -> flush(f), #Force printing to file regularly for progress monitoring
            CallbackFunction = cb_func,
            CallbackInterval=RP.t2/50,
        );


    end

    return

end


end