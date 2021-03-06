
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


function ss_current(model::OQSmodel; ss=steady_state(model), single_ex_tol=0.5, warn=true)::Float64

    Pss = populations(ss)

    # Check populations are physically valid
    tol = model.options.nonphysical_tol
    any(map(p -> !(-tol < p < 1+tol), real(diag(ss.data)))) && throw(OQSmodels.NonPhysicalStateError(Pss))

    # Check that single excitation approx is still justified and warn if not
    if Pss[end] < single_ex_tol
        inj_rate = model.env_processes.inject_1.spectrum.args.γ
        err_str = "Steady state population of ground state = $(Pss[end])\n"
        err_str *= "--> single excitation approximation possibly inaccurate? (Try lowering γ_inj from current value of $(inj_rate))"
        model.Ham.num_chains > 1 && (err_str *= "\nCurrent interchain_coupling = $(get_interchain_coupling(model))")
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

        (warn && Iss < 1e-12) && @warn "Steady state current = $(round(Iss, sigdigits=3)) ---> further calculations based on this value may be numerically unstable."

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


function vary_dipole_orientation!(m::OQSmodel, site, dx, dy, dz; update_H=true, update_L=true)

    m.Ham.coupling_func != full_dipole_coupling && error("Varying dipole orientation makes no sense when coupling function is not 'full_dipole_coupling'")

    #Update system Hamiltonian
    m.Ham.param_dict["dx$(site)"] = dx
    m.Ham.param_dict["dy$(site)"] = dy
    m.Ham.param_dict["dz$(site)"] = dz
    update_H && (m.Ham = update_H!(m.Ham)) #Recalculate H with updated param_dict

    #Update collective decay env processes
    m.env_processes[Symbol("rad_decay_x")].weightings[site] = dx
    m.env_processes[Symbol("rad_decay_y")].weightings[site] = dy
    m.env_processes[Symbol("rad_decay_z")].weightings[site] = dz

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
    dipole_θs = dipole_angles[1:2:end]
    dipole_ϕs = dipole_angles[2:2:end]

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


get_dipole_components(Ham, site::Int) = SVector(Ham.param_dict["dx$(site)"], Ham.param_dict["dy$(site)"], Ham.param_dict["dz$(site)"])
get_dipole_components(m::OQSmodel, site::Int) = get_dipole_components(m.Ham, site)
get_dipole_components(m::OQSmodel) = [get_dipole_components(m, s) for s in 1:numsites(m)]


get_dipole_sphericals(m::OQSmodel, site::Int) = SphericalFromCartesian()(get_dipole_components(m, site))

get_dipole_angles(m::OQSmodel, site::Int) = (d = get_dipole_sphericals(m, site); [d.θ, d.ϕ]) #Don't need to use modulo here since CoordinateTransformations.jl will enforce first quadrant constraints internally

get_dipole_angles(m::OQSmodel) = [get_dipole_angles(m, site) for site in 1:numsites(m)]


function vary_dipole_θ!(m, site, θ; kwargs...)
    d = get_dipole_sphericals(m, site) #Get existing dipole orientation
    dx, dy, dz = CartesianFromSpherical()(Spherical(d.r, θ, d.ϕ)) #Get cartesian components of new dipole orientation
    return vary_dipole_orientation!(m, site, dx, dy, dz; kwargs...)
end

#Non-mutating version
vary_dipole_θ(m, site, θ; kwargs...) = vary_dipole_θ!(copy(m), site, θ; kwargs...)

function vary_dipole_ϕ!(m, site, ϕ; kwargs...)
    d = get_dipole_sphericals(m, site) #Get existing dipole orientation
    dx, dy, dz = CartesianFromSpherical()(Spherical(d.r, d.θ, ϕ)) #Get cartesian components of new dipole orientation
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




# -------------------------------------------------------------------------------------------------------------------- #
#                                               Useful analysis utilities                                              #
# -------------------------------------------------------------------------------------------------------------------- #

get_decay_ops(m::OQSmodel) = m.Ham.coupling_func == full_dipole_coupling ? oper.([m.env_processes[Symbol("rad_decay_$a")] for a in ["x", "y", "z"]]) : oper.([m.env_processes.rad_decay])

export eigenstate_brightness
function eigenstate_brightness(m::OQSmodel)
    _, eigstates = eigenstates(m.Ham.op) #Drop ground state (idx 1)
    gs = nlevelstate(basis(m.Ham), get_env_state(m.Ham, "ground").idx)
    # return [sum(abs2(dagger(gs) * op * st) for op in get_decay_ops(m)) for st in eigstates[2:end]]
    return [abs2(dagger(gs) * sum(get_decay_ops(m)) * st) for st in eigstates[2:end]]
end

export eigen_populations
function eigen_populations(m::OQSmodel; ss=steady_state(m).data) 
    U = eigvecs(Array(m.Ham.op.data))
    return [(dot(v, ss, v)) for v in eachcol(U)]
end

export ss_emission_rates
function ss_emission_rates(m::OQSmodel; kwargs...)

    #Assume all radiative decay processes have the same rate
    Ks = string.(keys(m.env_processes))
    # k = m.Ham.coupling_func == distance_only_coupling ? :decay : Symbol(Ks[findfirst(occursin.("rad_decay_", Ks))])
    k = Symbol(Ks[findfirst(occursin.("rad_decay", Ks))])
    rate = m.env_processes[k].spectrum.args.rate

    # return rate * eigenstate_brightness(m) .* normalize!(eigen_populations(m; kwargs...)[2:end])
    return rate * eigenstate_brightness(m) .* eigen_populations(m; kwargs...)[2:end]
end


# -------------------------------------------------------------------------------------------------------------------- #
#                                               Useful plotting functions                                              #
# -------------------------------------------------------------------------------------------------------------------- #

import PyPlot as plt
using PyCall
# In-module workaround needed due to everything in python being a pointer (see https://github.com/JuliaPy/PyCall.jl#using-pycall-from-julia-modules)
const mpl = PyNULL()
function __init__()
    copy!(mpl, pyimport("matplotlib"))
end

function plot_site_energies_pyplot!(ax, Es::AbstractArray; Iss=nothing, pops=nothing, dx=0.4, dy=0.1)
    colours = pops === nothing ? "k" : (pops .- minimum(pops)) / (maximum(pops) - minimum(pops)) 
    my_cmap = plt.LinearSegmentedColormap.from_list("test", ["blue", "orange"], N=256)
    rect_verts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    out = ax.scatter(1:length(Es), Es, marker=rect_verts, s=5000/length(Es), c=colours, cmap=my_cmap) #"winter")
    ax.set_ylabel("Site Energy (eV)", size=12)
    Iss !== nothing && ax.set_title("\$ I_{ss} = $(round(Iss, sigdigits=3)) \$", size=12)
    return out
end

#Version which calculates pops from model
function plot_site_energies_pyplot!(ax, m::OQSmodel; kwargs...)
    ss_pops = populations(steady_state(m))[1:numsites(m)]
    return plot_site_energies_pyplot!(ax, site_energies(m); Iss=ss_current(m), pops=ss_pops, kwargs...)
end


select_marker(x) = real(x) > 0 ? "^" : "v" #Selects upwards or downward triangle to denote eigenstate phase

function plot_eigenstate_structure_pyplot!(axs, m::OQSmodel; ss=steady_state(m), log_brightness=true) #Can provide pre-calulated steady state as kwarg for efficiency
   
    ax1, ax2 = axs #Unpack axis pair
    #Calc some relevant physical quantities
    N = numsites(m)
    H_eigvals, H_eigenstates = eigenstates(dense(m.Ham.op)) 
    H_eigenstates = H_eigenstates[2:end] #Drop ground states (assuming they're the lowest energy states)
    H_eigvals = H_eigvals[2:end]
        
    # ss = steady_state(m) #Moved to kwarg
    eb_pops = [real(dagger(st) * ss * st) for st in H_eigenstates]

    brightness = eigenstate_brightness(m)
    log_brightness && (brightness = log10.(brightness))
    scaled_brightness = (brightness .- minimum(brightness)) / (maximum(brightness) - minimum(brightness))
    brightness_cmap = plt.LinearSegmentedColormap.from_list("brightnesses", ["k", "yellow"], N=256) #plt.cm.autumn #cgrad([:black, :yellow])
    
    #Add main scatter points
    for i in 1:N
        overlaps = [abs2(dagger(H_eigenstates[i])*nlevelstate(basis(m), j)) for j in 1:numsites(m)]
        real(H_eigenstates[i].data[1]) < 0 && (H_eigenstates[i] *= -1) #Fix first component as +ve to avoid arb. relative phase 'flip-flopping'
        phase_markers = select_marker.(H_eigenstates[i].data[1:N])
        colour = brightness_cmap(scaled_brightness[i])
        for j in 1:N #Need second loop to get different (up/down triangle) markers
            ax1.scatter([j], [i], marker=phase_markers[j], s=200*overlaps[j], color=colour, zorder=10, edgecolor="k")
        end
    end
    ax1.set_xlim(ax1.get_xlim() .- (0.08*numsites(m), 0)) #Aligns with top plot and makes space for energy spectrum
    ax1.set_xticks(1:N)
    ax1.set_yticks(1:N)
    ax1.set_xticklabels(["1", fill("", N-2)..., "$N"])
    xmin, xmax = ax1.get_xlim()
    for i in 1:N #Manually add partial grid
        ax1.axvline(i, c="grey", lw=0.5, ls="--", alpha=0.5, zorder=1)
        ax1.plot([1, xmax], [i, i], c="grey", lw=0.5, ls="--", alpha=0.5, zorder=1)
    end

    #Add eigenenergy spectrum
    d = (1 - xmin) #Distance from left edge of plot to start of eigenstate section
    scaled_spectrum = (N-1)*(H_eigvals .- minimum(H_eigvals)) / (maximum(H_eigvals) - minimum(H_eigvals)) .+ 1
    for i in 1:N
        colour = brightness_cmap(scaled_brightness[i])
        #Add energy value
        ax1.plot(
            [xmin + 0.35*d, xmin + 0.65*d], fill(scaled_spectrum[i], 2),
            lw=50/N, c=colour, solid_capstyle="round",
#             zorder=0.15 + 1e-2*i
        )
        #Add line connecting to eigenstate
        ax1.plot([xmin+0.65d, 1], [scaled_spectrum[i], i], lw=0.5, ls=":", c="grey", zorder=0.1)
    end
    #Add y-axis labels
    ax1.set_yticklabels([string(round(H_eigvals[1], sigdigits=3)), fill("", N-2)..., string(round(H_eigvals[end], sigdigits=3))])
    ax1.set_ylabel("Hamiltonian eigenenergy (eV)", labelpad=-20, size=12)
    
    #Add horizontal bar plot of eigenstate populations
    colours = [brightness_cmap(x) for x in scaled_brightness]
    ax2.barh(1:length(eb_pops), eb_pops, color=colours, edgecolor="k", linewidth=1)    
    ax2.set_xlabel("Eigenstate population", size=12, labelpad=12)
    ax2.set_yticks([])
    ax1.set_ylim(ax2.get_ylim()) #Align y axes
    
    #Add brightness colourbar
    sm = plt.cm.ScalarMappable(cmap=brightness_cmap)
    cbar = plt.colorbar(sm, ax=ax2, ticks=[0, 1])
    if log_brightness
        cbar.ax.set_yticklabels(string.(round.(extrema(brightness), sigdigits=3)))
        cbar.ax.set_ylabel("Logarithmic eigenstate brightness", labelpad=-15, size=10)
    else
        cbar.ax.set_yticklabels(["0", "$(round(maximum(brightness), sigdigits=3))"])
        cbar.ax.set_ylabel("Eigenstate brightness", labelpad=-15, size=10)
    end

end


export summarize_transport_model_pyplot
function summarize_transport_model_pyplot(m::OQSmodel; size=(10, 8), log_brightness=true, kwargs...)    
    
    #Initial layout setup
    fig = plt.figure(figsize=size)
    gs = mpl.gridspec.GridSpec(10, 16)
    ax1 = fig.add_subplot(py"""$(gs)[0:3, 1:12]""")
    ax1.set_xticks([])

    #Plot site energies on top subplot
    out = plot_site_energies_pyplot!(ax1, m; kwargs... )
    cax1 = fig.add_subplot(py"""$(gs)[0:3, 12:13]""")
    cbar1 = plt.colorbar(out, cax=cax1, ticks=[0, 1])
    cbar1.ax.set_yticklabels(["0", "Max"])
    cbar1.ax.set_ylabel("Site Population", labelpad=-15, size=10)
        
    ax2 = fig.add_subplot(py"""$(gs)[3:, 0:12]""")
    ax3 = fig.add_subplot(py"""$(gs)[3:, 12:]""")
    plot_eigenstate_structure_pyplot!([ax2, ax3], m; log_brightness=log_brightness)
        
#     plt.tight_layout()
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    return fig, [ax1, ax2, ax3]
end
