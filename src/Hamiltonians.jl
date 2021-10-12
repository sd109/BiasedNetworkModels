

# ---------------------------------------------------------------------------- #
#                                     Plan                                     #
# ---------------------------------------------------------------------------- #

# Implement a new sub-type of OQSmodel.SystemHamiltonian for 2D sheet-like transport networks

# The param dict for this Hamiltonian should contain:
#     -> On-site energies 
#     -> Real-space site positions (x, y, z)
#     -> Dipole orientations (|d|, θ & ϕ)


# We can then write various helper functions to dispatch on this type


mutable struct BiasedNetworkHamiltonian{F <: Any, O <: DataOperator} <: SystemHamiltonian
    chain_length::Int
    num_chains::Int
    coupling_func::F
    param_dict::Dict{String, Float64}
    env_states::Vector{EnvState}
    op::O
    geom_type::Symbol #e.g. :sheet, :prism, :ring1, :ring2 (for sanity checks in various other functions)
end



# -------------------------------------------------------------------------------------------------------------------- #
#                                                     Constructors                                                     #
# -------------------------------------------------------------------------------------------------------------------- #




# ------------------------------- Top level methods for setting up different geometries ------------------------------ #


function BiasedSheetHamiltonian(chain_length::Int, num_chains::Int, dE::Real;
     interchain_coupling=1.0, E0=100.0, coupling_func=full_dipole_coupling,
     dipole_orientations = [SVector(0, 0, 1) for i in 1:chain_length*num_chains] #Default is all dipoles parallel and pointing out of x-y plane
    )

    #Sanity check
    if coupling_func != full_dipole_coupling && dipole_orientations != [SVector(0, 0, 1) for i in 1:chain_length*num_chains]
        @warn "It looks like you have provided some non-default dipole orientations with a coupling function which is independent of dipole orientation..."
    end

    #Get Es from energy gradient dE
    Es = repeat(Float64[E0 + (chain_length-i)*dE for i in 1:chain_length], outer=num_chains)

    #Store each real pos (x, y, z) and dipole orientation (x, y, z) as StaticArrays.SVector instances for efficiency
    #(First 'for' loop is actually run second in for list comprehensions in Julia, just like python)
    site_pos = [SVector(i, j/cbrt(interchain_coupling), 0) for j in 0:num_chains-1 for i in 0:chain_length-1]

    return BiasedNetworkHamiltonian(chain_length, num_chains, Es, site_pos, coupling_func, dipole_orientations, :sheet)
end



function BiasedPrismHamiltonian(chain_length::Int, num_chains::Int, dE::Real;
    interchain_coupling=1.0, E0=100.0, coupling_func=full_dipole_coupling,
    dipole_orientations = [SVector(0, 0, 1) for i in 1:chain_length*num_chains] #Default is all dipoles parallel and pointing out of x-y plane
   )

    #Sanity checks
    num_chains < 3 && error("Prism geometry doesn't make sense for just $(num_chains) chain(s). Use BiasedSheetModel instead.")

    if coupling_func != full_dipole_coupling && dipole_orientations != [SVector(0, 0, 1) for i in 1:chain_length*num_chains]
        @warn "It looks like you have provided some non-default dipole orientations with a coupling function which is independent of dipole orientation..."
    end

    #Get Es from energy gradient dE
    Es = repeat([E0 + (chain_length-i)*dE for i in 1:chain_length], outer=num_chains)

    #Work out site-positions based on desired inter-chain coupling
    R = R_from_nn_dist(num_chains, 1/cbrt(interchain_coupling)) #Get prism 'radius' from inter-chain coupling (from utilities.jl)
    site_pos = [CartesianFromCylindrical()(Cylindrical(R, θ, z)) for θ in range(2π/num_chains, 2*π, length=num_chains) for z in 0:chain_length-1 ] #Second 'for' is done first

    return BiasedNetworkHamiltonian(chain_length, num_chains, Es, site_pos, coupling_func, dipole_orientations, :prism)
end



function BiasedRingHamiltonian(num_sites::Int, dE::Real, geom_type::Symbol;
     E0=100.0, coupling_func=full_dipole_coupling, 
     dipole_orientations = [SVector(0, 0, 1) for i in 1:num_sites] #Default is all dipoles parallel and pointing out of x-y plane
    )

    #Sanity checks
    isodd(num_sites) && error("Ring geometries must have an even number of sites.")

    if coupling_func != full_dipole_coupling && dipole_orientations != [SVector(0, 0, 1) for i in 1:num_sites]
        @warn "It looks like you have provided some non-default dipole orientations with a coupling function which is independent of dipole orientation..."
    end
    
    #Construct Es from energy gradient
    if geom_type == :ring1
        half_ring_Es = [E0 + dE*(num_sites÷2-i) for i in 0:num_sites÷2]
        Es = [half_ring_Es..., reverse(half_ring_Es[2:end-1])...]
    elseif geom_type == :ring2
        #Create energy gradient in half of sites then mirror in other half
        half_ring_Es = [E0 + dE*(num_sites÷2-i) for i in 1:num_sites÷2]
        Es = [half_ring_Es..., reverse(half_ring_Es)...]
    else
        error("Geometry type must be either :ring1 or :ring2 for BiasedRingHamiltonian.") #Sanity check
    end

    #Create site positions
    R = R_from_nn_dist(num_sites, 1)
    site_pos = [CartesianFromCylindrical()(Cylindrical(R, θ, 0)) for θ in range(2π/num_sites, 2*π, length=num_sites)]

    return BiasedNetworkHamiltonian(num_sites, 1, Es, site_pos, coupling_func, dipole_orientations, geom_type)
end



# function BiasedEllipseHamiltonian()
# end


# function BiasedMultiRingHamiltonian()
# end





# ------------------------------------ Main BiasedNetworkHamiltonian constructors ------------------------------------ #


# Constructor which creates param dict
function BiasedNetworkHamiltonian(
        chain_length::Int, num_chains::Int, 
        Es::Vector{E} where E <: Real, 
        site_pos::Vector, 
        coupling_func::Any,
        dipole_orientations::Vector, #{AbstractVector{D}} where D <: Real, 
        geom_type::Symbol
    )

    num_sites = chain_length * num_chains
    H = zeros(num_sites+1, num_sites+1)

    H[diagind(H)[1:end-1]] = Es #Add energies to H

    #Add inter-site couplings
    for (i, j) in subsets(1:num_sites, 2)
        V_ij = coupling_func(site_pos[i], site_pos[j], dipole_orientations[i], dipole_orientations[j])
        H[i, j] = V_ij
        H[j, i] = V_ij
    end

    #Construct explicit param dict for finite-diff compatibility
    param_dict = Dict{String, Float64}()
    for i in 1:num_sites
        param_dict["E$i"] = Es[i]
        param_dict["x$i"] = site_pos[i][1]
        param_dict["y$i"] = site_pos[i][2]
        param_dict["z$i"] = site_pos[i][3]
        param_dict["dx$i"] = dipole_orientations[i][1]
        param_dict["dy$i"] = dipole_orientations[i][2]
        param_dict["dz$i"] = dipole_orientations[i][3]
    end

    #Construct ground state
    ground_state = EnvState("ground", 0.0, num_sites+1)
    param_dict["E_ground"] = ground_state.energy
    param_dict["ground_idx"] = ground_state.idx

    #Convert Hamiltonian to QuantumOptics type
    op = Operator(NLevelBasis(num_sites+1), H)

    return BiasedNetworkHamiltonian(chain_length, num_chains, coupling_func, param_dict, EnvState[ground_state], op, geom_type) #Call base constructor method
end



#Constructor which operators solely on param dict to construct Hamiltonian operator
function BiasedNetworkHamiltonian(p::Dict{String, Float64}, chain_length::Int, num_chains::Int, coupling_func::Any, geom_type::Symbol)

    #Extract site energies, positions and dipole orientations from param_dict
    num_sites = chain_length * num_chains

    #Initialize new Hamiltonian operator 
    H = zeros(num_sites+1, num_sites+1)

    #Add energies to H
    for i in 1:num_sites
        H[i, i] = p["E$i"]
    end

    #And inter-site couplings
    for (i, j) in subsets(1:num_sites, 2)
        pos1 = [p["x$i"], p["y$i"], p["z$i"]]
        pos2 = [p["x$j"], p["y$j"], p["z$j"]]
        d1 = [p["dx$i"], p["dy$i"], p["dz$i"]]
        d2 = [p["dx$j"], p["dy$j"], p["dz$j"]]
        V_ij = coupling_func(pos1, pos2, d1, d2)
        H[i, j] = V_ij
        H[j, i] = V_ij
    end

    #Convert H to QuantumOptics operator 
    op = Operator(NLevelBasis(num_sites+1), H)

    #Construct ground state from param dict
    ground_state = EnvState("ground", p["E_ground"], p["ground_idx"])

    #Call base constructor
    return BiasedNetworkHamiltonian(chain_length, num_chains, coupling_func, p, EnvState[ground_state], op, geom_type)
end

#For finite-diff compatibility with OQSmodels/ModelCalculus.jl
SystemHamiltonian(H::BiasedNetworkHamiltonian) = BiasedNetworkHamiltonian(H.param_dict, H.chain_length, H.num_chains, H.coupling_func, H.geom_type)




# -------------------------------------------------------------------------------------------------------------------- #
#                                                   Useful functions                                                   #
# -------------------------------------------------------------------------------------------------------------------- #

# Extend some OQSmodel methods
OQSmodels.numsites(H::BiasedNetworkHamiltonian) = H.chain_length * H.num_chains
OQSmodels.site_energies(H::BiasedNetworkHamiltonian) = [H.param_dict["E$i"] for i in 1:numsites(H)]
OQSmodels.site_positions(H::BiasedNetworkHamiltonian) = vcat([[H.param_dict["x$i"] H.param_dict["y$i"] H.param_dict["z$i"]] for i in 1:numsites(H)]...)

#Method overload for faster copying
Base.copy(H::BiasedNetworkHamiltonian) = BiasedNetworkHamiltonian(H.chain_length, H.num_chains, H.coupling_func, copy(H.param_dict), deepcopy(H.env_states), copy(H.op), H.geom_type)



# Specific method for varying H params (more efficient than generic fallback in OQSmodels/ModelHamiltonians.jl)
function OQSmodels.vary_Hamiltonian_param!(Ham::BiasedNetworkHamiltonian, name::String, val::Real)

    Ham.param_dict[name] = val #Update param dict entry
    if match(r"E\d+", name) !== nothing #Site energy variation
        site = parse(Int, name[2:end])
        Ham.op.data[site, site] = val
    elseif match(r"E_.+", name) !== nothing #Env energy variation
        idx = Ham.param_dict["$(name[3:end])_idx"]
        Ham.op.data[idx, idx] = val

    # elseif # Add more cases here as required
    
    else #Generic (slow) fallback which creates new Ham instance and sets appropriate field
        new_H = vary_Hamiltonian_param(Ham, name, val)
        for f in fieldnames(BiasedNetworkHamiltonian)
            setfield!(Ham, f, getfield(new_H, f))
        end
    end

    return Ham
end



# function vary_interchain_coupling(H::BiasedSheetHamiltonian, new_val::Real)

# en