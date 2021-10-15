

# -------------------------------------------------------------------------------------------------------------------- #
#                                            Construction of actual OQSmodel                                           #
# -------------------------------------------------------------------------------------------------------------------- #


# Convenience method for sheet models
function BiasedSheetModel(chain_length, num_chains, dE;
    interchain_coupling=1.0, E0=100.0, coupling_func=full_dipole_coupling, 
    dipole_orientations = [SVector(0, 0, 1) for i in 1:chain_length*num_chains], #Default is all dipoles parallel and pointing out of x-y plane
    kwargs... #kwargs are passed to BiasedNetworkModel
   )

    H = BiasedSheetHamiltonian(chain_length, num_chains, dE; interchain_coupling=interchain_coupling, coupling_func=coupling_func, dipole_orientations=dipole_orientations, E0=E0)

    return BiasedNetworkModel(H; inj_sites=1:chain_length:num_chains*chain_length, ext_sites = chain_length:chain_length:num_chains*chain_length, kwargs...)
end


# Convenience method for prism models
function BiasedPrismModel(chain_length, num_chains, dE;
    interchain_coupling=1.0, E0=100.0, coupling_func=full_dipole_coupling, 
    dipole_orientations = [SVector(0, 0, 1) for i in 1:chain_length*num_chains], #Default is all dipoles parallel and pointing out of x-y plane
    kwargs... #kwargs are passed to BiasedNetworkModel
   )

    H = BiasedPrismHamiltonian(chain_length, num_chains, dE; interchain_coupling=interchain_coupling, coupling_func=coupling_func, dipole_orientations=dipole_orientations, E0=E0)

    return BiasedNetworkModel(H; inj_sites=1:chain_length:num_chains*chain_length, ext_sites = chain_length:chain_length:num_chains*chain_length, kwargs...)
end


# Convenience method for ring models
function BiasedRingModel(num_sites, dE, geom_type;
    E0=100.0, coupling_func=full_dipole_coupling, 
    dipole_orientations = [SVector(0, 0, 1) for i in 1:num_sites], #Default is all dipoles parallel and pointing out of x-y plane
    kwargs... #kwargs are passed to BiasedNetworkModel
   )

    #Deduce injection and extraction sites based on geom_type
    inj_sites = geom_type == :ring1 ? [1] : [1, num_sites]
    ext_sites = geom_type == :ring1 ? [num_sites÷2 + 1] : [num_sites÷2, num_sites÷2 + 1]

    H = BiasedRingHamiltonian(num_sites, dE, geom_type; coupling_func=coupling_func, dipole_orientations=dipole_orientations, E0=E0)

    return BiasedNetworkModel(H; inj_sites=inj_sites, ext_sites=ext_sites, kwargs...)
end



#Main model constructor
function BiasedNetworkModel(H;
        model_options = ModelOptions(err_on_nonphysical=false, finite_diff_func=finite_difference_derivative, extrapolate_deriv=false), #Make sure to use fast FiniteDiff derivatives
        ME_type = PauliME(),
        #Other kwargs are env rates and params
        inj_sites = [], #Dependent on Hamiltonian type so set by caller
        ext_sites = [], #Dependent on Hamiltonian type so set by caller
        γ_phonon = 1e-3,
        Γ_phonon = 0.5,
        T_phonon = 2.5875, # = 300 K if NN coupling is 10 meV
        ω0 = nothing, #Calcuated based on eigen-energy differences by default
        γ_inj = 1e-6,
        γ_ext = 3e-2,
        γ_decay = 2e-2,
        simple_spectra = true,
        T_cold = 0, # Only used if simple_spectra = false 
        T_hot = 20*2.5875, # Only used if simple_spectra = false 
        rescale_inj = true,
        rescale_ext = false,
        rescale_decay = false,
        eigen_inj = false,
        eigen_ext = false,
    )


    #Sanity check on eigen-energies - if inter-site couplings get too strong then eigenenergies can become < 0 which messes up injection / extraction processes
    eigen_Es, U = eigen(Array(H.op.data))
    if minimum(eigen_Es) < 0
        error("It looks like some system eigen-energies are < E_ground. This will lead to incorrect physics in the extraction processes.\nTry increasing E0 value or decreasing coherent coupling\nEigen-energies: $(round.(eigen_Es, sigdigits=4))")
    end

    #Rescale selected rates based on number of chains
    rescale_inj && (γ_inj /= H.num_chains)
    rescale_ext && (γ_ext /= H.num_chains)
    rescale_decay && (γ_decay /= H.num_chains)


    ### Env setup
    ground = get_env_state(H, "ground") 
    b = basis(H)

    #=
    Here, we want to account for the effects of dipole orientations on radiative decay rates.
    I think the easiest way to do this is to create a separate generator for decays and then modify the matrix elements individually.
    =#
    if H.coupling_func == full_dipole_coupling

        spectral_func = simple_spectra ? Sw_flat_down : Sw_flat
        decay_procs = EnvProcess[InteractionOp("decay_$i", transition(Float64, b, ground.idx, i)+transition(Float64, b, i, ground.idx), SpectralDensity(spectral_func, (T=T_cold, rate=γ_decay))) for i in 1:numsites(H)]
        L_decay = transport_generator(H.op, decay_procs, ME_type)    
    
        #Weight eigenstate transitions by their collective transition dipole moments
        site_dipoles = [get_dipole_components(H, site) for site in 1:numsites(H)]
        eigstates = eachcol(U[1:end-1, 2:end]) #Drop ground state
        weightings = [norm(sum(v .* site_dipoles))^2 for v in eigstates]
        idxs = diagind(L_decay)[2:end]
        L_decay[idxs] .*= weightings
        L_decay[1, 2:end] .*= weightings

    else
        #Single collective decay
        decay_op = sum(transition(Float64, b, ground.idx, i) + transition(Float64, b, i, ground.idx) for i in 1:numsites(H)) #Collective decay op
        spectral_func = simple_spectra ? Sw_flat_down : Sw_flat
        decay_procs = [InteractionOp("decay", decay_op, SpectralDensity(spectral_func, (T=T_cold, rate=γ_decay)))]
        L_decay = transport_generator(H.op, decay_procs, ME_type)

    end
    @show L_decay

    #Phonons
    deph_ops = OQSmodels.site_dephasing_ops(H)
    ω0 === nothing && (ω0 = mean(diff(eigen_Es)))
    # Sw = SpectralDensity(rescaled_Sw_drude_lorentz, (T=T_phonon, Γ=Γ_phonon, ω0=sqrt(dE_ratio^2 - Γ_phonon^2), γ=γ_phonon))
    Sw = SpectralDensity(rescaled_Sw_drude_lorentz, (T=T_phonon, Γ=Γ_phonon, ω0=ω0, γ=γ_phonon))
    deph_procs = [InteractionOp("phonons_$i", op, Sw) for (i, op) in enumerate(deph_ops)]

    #Injection processes
    spectral_func = simple_spectra ? Sw_flat_up : Sw_flat
    if eigen_inj
        op =  Operator(b, U * (transition(Float64, b, b.N, 1) + transition(Float64, b, 1, b.N)).data * inv(U)) # b.N is highest energy eigenstate and 1 is ground (eigen)state
        inj_procs = [InteractionOp("inject_eb", op, SpectralDensity(spectral_func, (T=T_hot, γ=γ_inj)))]
    else
        inj_procs = [InteractionOp("inject_$(site)", transition(Float64, b, site, ground.idx) + transition(Float64, b, ground.idx, site), SpectralDensity(spectral_func, (T=T_hot, γ=γ_inj))) for site in inj_sites]
    end

    #Extraction processes
    spectral_func = simple_spectra ? Sw_flat_down : Sw_flat
    if eigen_ext
        op = Operator(b, U * (transition(Float64, b, 1, 2) + transition(Float64, b, 2, 1)).data * inv(U)) # '2' is lowest energy system eigenstate and 1 is ground (eigen)state
        ext_procs = [InteractionOp("extract_eb", op, SpectralDensity(spectral_func, (T=T_cold, γ=γ_ext)))]
    else
        ext_procs = [InteractionOp("extract_$(site)", transition(Float64, b, ground.idx, site) + transition(Float64, b, site, ground.idx), SpectralDensity(spectral_func, (T=T_cold, γ=γ_ext))) for site in ext_sites]
    end

    #List of (non-radiative-decay) env processes
    other_env_procs = EnvProcess[
        deph_procs...,
        # decay_proc, 
        inj_procs...,
        ext_procs...,
    ]

    L_other = transport_generator(H.op, other_env_procs, ME_type)

    return OQSmodel(H, [other_env_procs..., decay_procs...], ME_type, ground.idx, options=model_options, L=L_decay+L_other) #Use ground as initial state

end





# #Main model constructor
# function BiasedNetworkModel(H;
#     model_options = ModelOptions(err_on_nonphysical=false, finite_diff_func=finite_difference_derivative, extrapolate_deriv=false), #Make sure to use fast FiniteDiff derivatives
#     ME_type = PauliME(),
#     kwargs... #Extra kwargs are env process parameters   
#     )

#     ground = get_env_state(H, "ground") 

#     L_other, L_decay = _transport_generators(H, ground; kwargs...)

#     return OQSmodel(H, env_processes, ME_type, ground.idx, options=model_options,)# L=zeros(4, 4)) #Use ground as initial state

# end



# function _env_processes(H, ground;
#     inj_sites = [], #Dependent on Hamiltonian type so set by caller
#     ext_sites = [], #Dependent on Hamiltonian type so set by caller
#     γ_phonon = 1e-3,
#     Γ_phonon = 0.5,
#     T_phonon = 2.5875, # = 300 K if NN coupling is 10 meV
#     ω0 = nothing, #Calcuated based on eigen-energy differences by default
#     γ_inj = 1e-6,
#     γ_ext = 3e-2,
#     γ_decay = 2e-2,
#     simple_spectra = true,
#     T_cold = 0, # Only used if simple_spectra = false 
#     T_hot = 20*2.5875, # Only used if simple_spectra = false 
#     rescale_inj = true,
#     rescale_ext = false,
#     rescale_decay = false,
#     eigen_inj = false,
#     eigen_ext = false,
# )

#     #Sanity check on eigen-energies - if inter-site couplings get too strong then eigenenergies can become < 0 which messes up injection / extraction processes
#     eigen_Es, U = eigen(Array(H.op.data))
#     if minimum(eigen_Es) < 0
#         error("It looks like some system eigen-energies are < E_ground. This will lead to incorrect physics in the extraction processes.\nTry increasing E0 value or decreasing coherent coupling\nEigen-energies: $(round.(eigen_Es, sigdigits=4))")
#     end

#     #Rescale selected rates based on number of chains
#     rescale_inj && (γ_inj /= H.num_chains)
#     rescale_ext && (γ_ext /= H.num_chains)
#     rescale_decay && (γ_decay /= H.num_chains)


#     ### Env setup
#     b = basis(H)

#     #Phonons
#     deph_ops = OQSmodels.site_dephasing_ops(H)
#     ω0 === nothing && (ω0 = mean(diff(eigen_Es)))
#     # Sw = SpectralDensity(rescaled_Sw_drude_lorentz, (T=T_phonon, Γ=Γ_phonon, ω0=sqrt(dE_ratio^2 - Γ_phonon^2), γ=γ_phonon))
#     Sw = SpectralDensity(rescaled_Sw_drude_lorentz, (T=T_phonon, Γ=Γ_phonon, ω0=ω0, γ=γ_phonon))
#     deph_procs = [InteractionOp("phonons_$i", op, Sw) for (i, op) in enumerate(deph_ops)]

#     #Collective decay
#     decay_op = sum(transition(b, ground.idx, i) + transition(Float64, b, i, ground.idx) for i in 1:numsites(H)) #Collective decay op
#     spectral_func = simple_spectra ? Sw_flat_down : Sw_flat
#     decay_proc = InteractionOp("decay", decay_op, SpectralDensity(spectral_func, (T=T_cold, rate=γ_decay)))

#     #Injection processes
#     spectral_func = simple_spectra ? Sw_flat_up : Sw_flat
#     if eigen_inj
#         op =  Operator(b, U * (transition(Float64, b, b.N, 1) + transition(Float64, b, 1, b.N)).data * inv(U)) # b.N is highest energy eigenstate and 1 is ground (eigen)state
#         inj_procs = [InteractionOp("inject_eb", op, SpectralDensity(spectral_func, (T=T_hot, γ=γ_inj)))]
#     else
#         inj_procs = [InteractionOp("inject_$(site)", transition(Float64, b, site, ground.idx) + transition(Float64, b, ground.idx, site), SpectralDensity(spectral_func, (T=T_hot, γ=γ_inj))) for site in inj_sites]
#     end

#     #Extraction processes
#     spectral_func = simple_spectra ? Sw_flat_down : Sw_flat
#     if eigen_ext
#         op = Operator(b, U * (transition(Float64, b, 1, 2) + transition(Float64, b, 2, 1)).data * inv(U)) # '2' is lowest energy system eigenstate and 1 is ground (eigen)state
#         ext_procs = [InteractionOp("extract_eb", op, SpectralDensity(spectral_func, (T=T_cold, γ=γ_ext)))]
#     else
#         ext_procs = [InteractionOp("extract_$(site)", transition(Float64, b, ground.idx, site) + transition(Float64, b, site, ground.idx), SpectralDensity(spectral_func, (T=T_cold, γ=γ_ext))) for site in ext_sites]
#     end

#     #Put all env processes together
#     env_procs = EnvProcess[
#         deph_procs...,
#         decay_proc,
#         inj_procs...,
#         ext_procs...,
#     ]

#         return env_procs
# end

