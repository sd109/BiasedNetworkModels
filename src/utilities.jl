
# -------------------------------------------------------------------------------------------------------------------- #
#                                      A collection of useful, reusable functions                                      #
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------- Hamiltonian geometry related functions -------------------------------------- #


# Distance dependent couplings (ignoring dipole orientations)
function distance_only_coupling(pos1, pos2, d1, d2)
    return 1/norm(pos2-pos1)^3
end

# Distance and orientation dependent dipole-dipole coupling 
function full_dipole_coupling(pos1, pos2, d1, d2)
    r = pos2-pos1
    r_norm = norm(r)
    return dot(d1, d2) / r_norm^3 - 3*dot(d1, r)*dot(d2, r) / r_norm^5
end

# Function for working out circle radius which gives the desired nearest neighbour distance
R_from_nn_dist(N, nn_dist) = sqrt(nn_dist^2/(2*(1-cos(2π/N))))




# ------------------------------------------------ Spectral densities ------------------------------------------------ #

Sw_drude_lorentz(w, T, Γ, ωc, γ) =  w == 0 ? 0 : π*γ*Γ*abs(w) / (Γ^2 + (abs(w) - ωc)^2) * (nbe(w, T) + (w > 0))

Sw_flat(w, T, rate) = w == 0 ? 0 : rate * (nbe(w, T) + (w > 0))

Sw_flat_up(ω, T, rate) = rate * (ω < 0)

Sw_flat_down(ω, T, rate) = rate * (ω > 0)

rescaled_nbe(w, T) = ( exp(abs(w) / T) - 1 )^-1.0

rescaled_Sw_drude_lorentz(w, T, Γ, ωc, γ) =  w == 0 ? 0 : π*γ*Γ*abs(w) / (Γ^2 + (abs(w) - ωc)^2) * (rescaled_nbe(w, T) + (w > 0))



# -------------------------------------------- Transport-related functions ------------------------------------------- #


function ss_current(model::OQSmodel; ss=steady_state(model), single_ex_tol=0.5)::Float64

    Pss = populations(ss)

    # Check populations are physically valid
    tol = model.options.nonphysical_tol
    any(map(p -> !(-tol < p < 1+tol), real(diag(ss.data)))) && throw(OQSmodels.NonPhysicalStateError(Pss))

    # Check that single excitation approx is still justified and warn if not
    if Pss[end] < single_ex_tol
        inj_rate = model.env_processes.inject_1.spectrum.args.γ
        err_str = "Steady state population of ground state = $(Pss[end])\n"
        err_str *= "--> single excitation approximation possibly inaccurate? (Try lowering γ_inj from current value of $(inj_rate))"
        err_str *= "\nCurrent interchain_coupling = $(get_interchain_coupling(model))"
        if model.options.err_on_nonphysical
            throw(error(err_str))
        else
            @warn err_str
        end
    end

    # Calculate the current
    if :extract_eb in keys(model.env_processes) # model must be using eigenbasis extraction (eigen_ext=true in constructor)
        #Transform steady state to eigenbasis
        U = eigvecs(Array(model.Ham.op.data))
        ss_eb = inv(U) * ss.data * U
        P = populations(ss_eb)[2] #Get population of lowest energy system eigenstate
        return P * model.env_processes.extract_eb.spectrum.args.γ

    else
        ext_procs = [p for p in model.env_processes if occursin("extract_", p.name)]
        Iss = mapreduce(+, ext_procs) do proc
            site = parse(Int, match(r"\d+", proc.name).match)
            rate = proc.spectrum.args.γ
            rate*Pss[site]            
        end

        Iss < 1e-12 && @warn "Steady state current = $(round(Iss, sigdigits=3)) ---> further calculations based on this value may be numerically unstable."

        return Iss #::Float64 #This branch is not type stable (idk why...) so annotate type <- do this at function line instead
    end
end




# -------------------------------------------------------------------------------------------------------------------- #
#                           Efficicent methods for varying multiple parameters simultaneously                          #
# -------------------------------------------------------------------------------------------------------------------- #


function vary_site_energies!(model::OQSmodel, new_Es, sites; update_H=true, update_L=true)
    
    length(sites) != length(new_Es) && error("Site list and energies must be the same length.") #Sanity check

    #Set new param dict entries
    for (site, E) in zip(sites, new_Es)
        model.Ham.param_dict["E$(site)"] = E
    end
    
    #Recalculate model fields if needed
    update_H && (model.Ham = update_H!(model.Ham))
    update_L && (model.L = transport_generator(model))

    return model
end

function vary_interchain_coupling!(model::OQSmodel, new_coupling_val; update_H=true, update_L=true)

    """ 
    Note - The actual inter-chain coupling wont be exactly `new_coupling_val` if full_dipole_coupling is used, 
    aside from in a few special cases (e.g. a sheet with all dipoles parallel and pointing out of plane)
    """

    #Sanity check
    if model.Ham.num_chains < 2 #|| !(model.Ham.geom_type == :sheet || model.Ham.geom_type == :prism)
        error("Varying the inter-chain coupling only makes sense for sheet and prism geometries with at least 2 chains")
    end

    #Get new positions of all sites
    if model.Ham.geom_type == :sheet
        new_site_pos = [SVector(i, j/cbrt(new_coupling_val), 0) for j in 0:model.Ham.num_chains-1 for i in 0:model.Ham.chain_length-1]
    elseif model.Ham.geom_type == :prism
        #Work out site-positions based on desired inter-chain coupling
        R = R_from_nn_dist(model.Ham.num_chains, 1/cbrt(new_coupling_val)) #Get prism 'radius' for desired inter-chain coupling (from utilities.jl)
        new_site_pos = [CartesianFromCylindrical()(Cylindrical(R, θ, z)) for θ in range(2π/model.Ham.num_chains, 2*π, length=model.Ham.num_chains) for z in 0:model.Ham.chain_length-1 ] #Second 'for' is done first
    else
        error("Varying the inter-chain coupling only makes sense for sheet and prism geometries with at least 2 chains")
    end

    #Update param dict entries
    for (i, (x, y, z)) in enumerate(new_site_pos)
        model.Ham.param_dict["x$i"] = x
        model.Ham.param_dict["y$i"] = y
        model.Ham.param_dict["z$i"] = z
    end

    #Recalculate model fields if needed
    update_H && (model.Ham = update_H!(model.Ham))
    update_L && (model.L = transport_generator(model))

    return model
end

#Non-mutating version
vary_interchain_coupling(model::OQSmodel, new_coupling_val; kwargs...) = vary_interchain_coupling!(copy(model), new_coupling_val; kwargs...)


function vary_dipole_orientation!(m, site, dx, dy, dz; update_H=true, update_L=true)
    m.Ham.param_dict["dx$(site)"] = dx
    m.Ham.param_dict["dy$(site)"] = dy
    m.Ham.param_dict["dz$(site)"] = dz
    update_H && (m.Ham = update_H!(m.Ham)) #Recalculate H with updated param_dict
    update_L && (m.L = transport_generator(m))
    return m
end


# ------------------------------------------------ Combination methods ----------------------------------------------- #

function vary_energies_and_interchain_coupling!(m::OQSmodel, x, sites)
    Es = x[1:end-1]
    new_coupling = x[end]
    vary_site_energies!(m, Es, sites, update_H=false, update_L=false)
    vary_interchain_coupling!(m, new_coupling, update_H=true, update_L=true) #Only recalculate H and L once
    return m
end


function vary_energies_and_orientations_and_coupling!(m::OQSmodel, x::Vector{R} where R <: Real, E_sites, dipole_sites)

    #Sanity check
    length(x) != length(E_sites) + 1 + 2*length(dipole_sites) && error("Length of x vector is incorrect.")

    L = length(E_sites)
    Es = x[1:L]
    new_coupling = x[L+1]
    dipole_angles = x[L+2:end]
    dipole_θs = x[1:2:end]
    dipole_ϕs = x[2:2:end]

    vary_site_energies!(m, Es, E_sites, update_H=false, update_L=false)
    vary_interchain_coupling!(m, new_coupling, update_H=false, update_L=false) 
    for (site, θ, ϕ) in zip(dipole_sites, dipole_θs, dipole_ϕs)
        vary_dipole_θ!(m, site, θ, update_H=false, update_L=false)
        vary_dipole_ϕ!(m, site, ϕ, update_H=false, update_L=false)
    end

    #Only recalculate H and L once
    m.Ham = update_H!(m.Ham)
    m.L = transport_generator(m)

    return m
end


# -------------------------------------------------------------------------------------------------------------------- #
#                                              New DiffParam constructors                                              #
# -------------------------------------------------------------------------------------------------------------------- #



# ------------------------------------------------- Chain separation ------------------------------------------------- #


function get_interchain_coupling(model::OQSmodel)
    model.Ham.num_chains < 2 && error("Can't get inter-chain coupling for a system with onln 1 chain")
    existing_pos = site_positions(model)
    chain_sep = norm(existing_pos[1, :] - existing_pos[model.Ham.chain_length+1, :])
    return 1 / chain_sep^3
end

# DiffParam constructor 
function InterchainCouplingDeriv(model::OQSmodel, scale::Real)
    coupling_val = get_interchain_coupling(model)
    return OQSmodels.DiffParam("Inter-chain coupling", coupling_val, vary_interchain_coupling, scale)
end



# --------------------------------------------------- Dipole angles -------------------------------------------------- #


get_dipole_components(m::OQSmodel, site::Int) = SVector(m.Ham.param_dict["dx$(site)"], m.Ham.param_dict["dy$(site)"], m.Ham.param_dict["dz$(site)"])

get_dipole_sphericals(m::OQSmodel, site::Int) = SphericalFromCartesian()(get_dipole_components(m, site))

# get_dipole_angles(m::OQSmodel, site::Int) = (d = get_dipole_sphericals(m, site); [d.θ % π, d.ϕ % π/2]) #Use modulo ensure we don't end up with arbitrarily large angles (CoordinateTransformations uses -π<θ<π & -π/2<ϕ<π/2)
get_dipole_angles(m::OQSmodel, site::Int) = (d = get_dipole_sphericals(m, site); [d.θ, d.ϕ]) #Don't need to use modulo here since CoordinateTransformations.jl will enforce first quadrant constraints internally

get_dipole_angles(m::OQSmodel) = [get_dipole_angles(m, site) for site in 1:numsites(m)]


function vary_dipole_θ!(m, site, θ; kwargs...)
    d = get_dipole_sphericals(m, site) #Get existing dipole orientation
    dx, dy, dz = CartesianFromSpherical()(Spherical(d.r, θ % π, d.ϕ)) #Get cartesian components of new dipole orientation (use % π to ensure perturbed angles remain in first quadrant)
    return vary_dipole_orientation!(m, site, dx, dy, dz; kwargs...)
end

#Non-mutating version
vary_dipole_θ(m, site, θ; kwargs...) = vary_dipole_θ!(copy(m), site, θ; kwargs...)

function vary_dipole_ϕ!(m, site, ϕ; kwargs...)
    d = get_dipole_sphericals(m, site) #Get existing dipole orientation
    dx, dy, dz = CartesianFromSpherical()(Spherical(d.r, d.θ, ϕ % π/2 )) #Get cartesian components of new dipole orientation (use % π/2 to ensure perturbed angles remain in first quadrant)
    return vary_dipole_orientation!(m, site, dx, dy, dz; kwargs...)
end
#Non-mutating version
vary_dipole_ϕ(m, site, ϕ; kwargs...) = vary_dipole_ϕ!(copy(m), site, ϕ; kwargs...)

#Dispatch on single site
function DipoleThetaDeriv(m::OQSmodel, site::Int, scale::Real)
    param_val = SphericalFromCartesian()(get_dipole_components(m, site)).θ
    return DiffParam("dθ$(site)", param_val, (m, θ) -> vary_dipole_θ(m, site, θ), scale)
end

#Dispatch on single site
function DipolePhiDeriv(m::OQSmodel, site::Int, scale::Real)
    param_val = SphericalFromCartesian()(get_dipole_components(m, site)).θ
    return DiffParam("dϕ$(site)", param_val, (m, ϕ) -> vary_dipole_ϕ(m, site, ϕ), scale)
end

#Dispatch on single site, both angles
DipoleAngleDeriv(m::OQSmodel, site::Int, scale::Real) = [DipoleThetaDeriv(m, site, scale), DipolePhiDeriv(m, site, scale)]

#Dispatch on multiple sites, both angles
DipoleAngleDeriv(m::OQSmodel, sites::Union{UnitRange{Int}, Vector}, scale::Real) = reduce(vcat, DipoleAngleDeriv(m, i, scale) for i in sites)

#Dispatch on all sites, both angles
DipoleAngleDeriv(m::OQSmodel, scale::Real) = DipoleAngleDeriv(m, 1:numsites(m), scale)

