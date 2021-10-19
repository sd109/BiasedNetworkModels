

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
    EnergyVarSites = nothing, #What's the best way to set these? - in separate command line app script (see RunMultiObjectiveOpt.jl)
    DipoleVarSites = nothing,
    SingleObjTimeLimit = 60,
    SingleObjMaxIters = 10^7, #Prefer to rely on time limit but set this as fallback
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

    return RP
end



# -------------------------------------------------------------------------------------------------------------------- #
#                                        Opimization-specific utility functions                                        #
# -------------------------------------------------------------------------------------------------------------------- #



function create_init_model(run_params)

    if run_params.Geom == :sheet
        return BiasedSheetModel(run_params.ChainLength, run_params.NumChains, run_params.dE_ratio; interchain_coupling=1/run_params.ChainSep^3, coupling_func=run_params.CouplingFunc)
    elseif run_params.Geom == :prism
        return BiasedPrismModel(run_params.ChainLength, run_params.NumChains, run_params.dE_ratio; interchain_coupling=1/run_params.ChainSep^3, coupling_func=run_params.CouplingFunc)
    elseif occursin("ring", string(run_params.Geom))
        return BiasedRingModel(run_params.ChainLength, run_params.dE_ratio, run_params.Geom; coupling_func=run_params.CouplingFunc)
    else
        error("Could not create starting model for geom: $(run_params.Geom)")
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
    run_params.WithSep && push!(x, 1/cbrt(get_interchain_coupling(model)))
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
        new_θs = new_dipole_angles[1:2:end] .% π #Make sure angles are within [-π, π]
        new_ϕs = new_dipole_angles[2:2:end] .% π/2 #Make sure angles are within [-π/2, π/2]
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

    # all_angles = reduce(vcat, [get_dipole_angles(model, s) for s in 1:numsites(model)])
    all_angles = reduce(vcat, get_dipole_angles(model))
    θs = all_angles[1:2:end] #.% π
    ϕs = all_angles[2:2:end] #.% π/2

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


#Makes sure all polar angles are within [-π, π] & all azimuth angles within [-π/2, π/2]
function fold_angles(x, RP)

    #If dipoles aren't included in x then just return to caller
    !(RP.WithDipoles) && return x

    new_x = copy(x) #Avoid mutating input vec
    idx = 1 #idx of first angle in x vec

    if RP.WithEs
        num_Es = RP.Symm ? length(RP.EnergyVarSites) ÷ RP.NumChains : length(RP.EnergyVarSites)
        idx += num_Es
    end 
    if RP.WithSep
        idx += 1
    end

    #Check idx looks correct
    @assert length(new_x[idx:end]) == 2*length(RP.DipoleVarSites)

    #Rescale angles
    new_x[idx:2:end] .%= π
    new_x[idx+1:2:end] .%= π/2

    return new_x
end


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


function current_and_QFIM_trace(model::OQSmodel, run_params::NamedTuple, I_ref::Real)#::Tuple{Float64, Float64}
    Iss = ss_current(model)
    QFIM = QuantumFIM(model, steady_state, FIM_params(model, run_params))::Matrix{ComplexF64} #Annotate return type
    return (-Iss/I_ref, real(tr(QFIM))) #Use -ve Iss since we want to maximize this value
end

#Version which sets new x before calc of objective values
current_and_QFIM_trace(m::OQSmodel, x::Vector{R} where R <: Real, RP::NamedTuple, I_ref::Real) = current_and_QFIM_trace(update_x(m, x, RP), RP, I_ref)


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


#For checking if single-obj optima are allowed to be added to initial MOO population
function in_search_range(x::Vector, RP::NamedTuple) 

    length(x) != length(RP.SearchRange) && error("length(x) = $(length(x)) but length(RP.SearchRange) = $(length(RP.SearchRange)) ===> Cannot check if x is within search range")

    min_vals = getindex.(RP.SearchRange, 1)
    max_vals = getindex.(RP.SearchRange, 2)
    if !all(min_vals .<= x .<= max_vals)
        #Print some extra info if x is outwith search range
        println("Elements that were within search range: ", getindex.(RP.SearchRange, 1) .< x .< getindex.(RP.SearchRange, 2))
        println("Elements: ", collect(zip(min_vals, x, max_vals)), "\n")
        return false
    end

    return true
end




# -------------------------------------------------------------------------------------------------------------------- #
#                                             Main optimization procedures                                             #
# -------------------------------------------------------------------------------------------------------------------- #


function run_ensemble_opt(model, run_params, I_ref; alg=ADAM(0.001), dis=1e-2)

    opt_func = OptimizationFunction((x, p) -> -1*ss_current(update_x!(p, x, run_params))/I_ref, GalacticOptim.AutoFiniteDiff())

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
        sol = solve(prob, alg, maxiters=run_params.SingleObjMaxIters, cb = (p, f_x) -> early_stopping_cb(p, f_x, run_params.SingleObjTimeLimit, start_time, obj_hist))
        flush(stdout) #Does this flush to file if redirect_stdout(f) has been called? I think so
        sol
    end

    #Wait for unperturbed opt to finish too, then add it to sol_list
    push!(sol_list, fetch(opt_task))

    return sol_list

end
#Version which calculates referece current automatically
run_ensemble_opt(m::OQSmodel, RP::NamedTuple; kwargs...) = run_ensemble_opt(m, RP, ss_current(m); kwargs...)



function run_multi_obj_opt(RP::NamedTuple; obj_func=current_and_QFIM_trace, trace_interval=600, kwargs...) #kwargs are passed to run_ensemble_opt

    #Check that we wont overwrite existing results
    any(isfile.(RP.SaveName .* ["-res.sjl", "-res-bak.txt", "-trace.txt", "-res.jld2"])) && error("File name already exists - move existing file or choose different save name.")

    start_model = create_init_model(RP)
    I_ref = ss_current(start_model)

    #Set search range for multi-obj opt automatically if it's not set
    if RP.SearchRange === nothing
        SR = Tuple{Float64, Float64}[] #Initialize vector of tuples
        if RP.WithEs 
            Es = site_energies(start_model)[RP.EnergyVarSites]
            RP.Symm && (Es = Es[1:length(RP.EnergyVarSites)÷start_model.Ham.num_chains])
            append!(SR, collect(zip(Es .- 5, Es .+ 5)))
        end
        if RP.WithSep
            push!(SR, (1/cbrt(20), 1/cbrt(1e-3))) #Express limits in terms of maximal allowed (dimensionless) coupling
        end
        if RP.WithDipoles
            append!(SR, repeat([(-π, π), (-π/2, π/2)], outer=length(RP.DipoleVarSites))) #Allow θs within [-π, π] and ϕs within [-π/2, π/2] (allow 10% outside either way)
            # append!(SR, repeat([ 1.1 .* (-π, π), 1.1 .* (-π/2, π/2)], outer=length(RP.DipoleVarSites))) #Allow θs within [-π, π] and ϕs within [-π/2, π/2] (allow 10% outside either way)
        end

        RP = (RP..., SearchRange = SR) #Store search range in run params
    end

    # Note start time and run params in trace file
    START_TIME = Dates.now()
    open("$(RP.SaveName)-trace.txt", "w") do f
        str = "\nStarting optimization run on $(Dates.Date(START_TIME)) at $(Dates.format(Dates.Time(START_TIME), "HH:MM:SS")) with run parameters:\n"
        for (k, v) in pairs(RP)
            str *= "  $k => $v\n"
        end
        write(f, str * "\n\n")
    end



    save_data = open("$(RP.SaveName)-trace.txt", "a") do f

        #Send all printed output to a text file
        redirect_stdout(f)
        #Send all warnings to the same text file
        logger = SimpleLogger(f)
        global_logger(logger)

        #Needs to be defined within 'open' block so that f is accesible
        function PF_callback(oc::BlackBoxOptim.OptRunController)
            println("Range of enhancement factors found so far: ", extrema(first.(getfield.(fitness.(pareto_frontier(oc.evaluator.archive)), :orig))), "\n")
            flush(f) #Force printing to file regularly for progress monitoring
        end
    
        # Perform single-obj ensemble opt
        sol_list = run_ensemble_opt(start_model, RP, I_ref; kwargs...)

        # Set up multi-obj opt run
        opt_ctrl = bbsetup(
            x -> obj_func(start_model, x, RP, I_ref), #Function to be optimized
            Method=:borg_moea,
            FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true), #, aggregator=f -> f[1]), #Use aggregator to display largest enhancement factor in trace (rather than equally weighted sum of both objectives)
            SearchRange=RP.SearchRange,
            # PopulationSize=100, #Default is 50
            # MaxTime=RP.t2,
            MaxSteps = RP.MaxSteps, #Is ignored by BlackBoxOptim if non-zero MaxTime is provide
            # MaxStepsWithoutProgress=10^5, #default is 10^4
            # MaxNumStepsWithoutFuncEvals = 5000, #Default is 100
            # ϵ=[1e-3, 1e-3], #Size of 'epsilon box' for measuring fitness progress
            ϵ=1e-5, #Size of 'epsilon box' for measuring fitness progress
            TraceInterval=trace_interval,
            CallbackFunction = PF_callback,
            CallbackInterval=trace_interval,
        );

        #Add ensemble opt results to starting population
        candidates = [fold_angles(sol.minimizer, RP) for sol in sol_list]
        viable_candidates = [c for c in candidates if in_search_range(c, RP)]
        a, b = length(viable_candidates), length(sol_list)
        if a < 0.1*b
            error("$(round(a/b*100))% of single optimization candidate are within the search range -> restart calculation with wider search range")
        else
            println("$a / $b candidate solutions from ensemble opt added to initial multi-obj population.")
        end

        #Start multi-obj optimization run
        println("\n\n\n Starting multi-objective optimization run:\n---------------------------------------------\n")
        flush(f)
        res = bboptimize(opt_ctrl, viable_candidates)


        #   Save results 
        ####################

        # Might as well sort PF here instead of during later analysis
        PF = pareto_frontier(res)
        EFs = -1*getindex.(fitness.(PF), 1)
        sorted_idxs = getindex.(sort(collect(zip(EFs, 1:length(EFs)))), 2)
        PF = PF[sorted_idxs]

        save_data = (
            start_model = start_model,
            run_params = RP,
            sol_list = sol_list,
            # single_obj_opts = [(sol.minimum, sol.minimizer) for sol in sol_list],
            pf = PF,
            MOO_starting_candidates = viable_candidates,
            # MOO_opt_params = res.parameters,
            MOO_stop_reason = res.stop_reason,
        )

        #Serialize relevant stuff as a named tuple
        serialize("$(RP.SaveName)-res.sjl", save_data)
        println("Results successfully serialized to: '$(RP.SaveName)-res.sjl'")

        #Add MOO params after serialize save (these don't serialize nicely between different minor Julia versions due to anon callback funcs)
        save_data = (save_data..., MOO_opt_params = res.parameters)

        # Save a back up as a text file
        open("$(RP.SaveName)-res-bak.txt", "w") do f_bak
            for (name, val) in pairs(save_data)
                println(f_bak, "\n[", name, "]\n", val, "\n")
            end
        end
        println("Backup results successfully written to: '$(RP.SaveName)-res-bak.txt'")

        # Save another backup using JLD2 just in case
        jldsave("$(RP.SaveName)-res.jld2", res=save_data)
        println("Additional backup results successfully saved to: '$(RP.SaveName)-res.jld2'")

        save_data #Return save data from 'do' block
    end

    return save_data #Return results from func as well as saving to files, just in case we want to work interactively
end


