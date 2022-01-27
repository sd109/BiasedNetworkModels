
using Pkg; Pkg.activate(".")

#Use ArgParse.jl to allow running batch jobs via a single bash script
using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin

    #Required args

    "--SaveName"
        help = "File name prefix for saving all results file (must not already exist)"
        arg_type = String
        required = true

    "--Geom"
        help = "Choose between sheet, prism, ring1 or ring2"
        arg_type = Symbol
        required = true

    "--ChainLength"
        help = "Number of sites per chain"
        arg_type = Int
        required = true

    "--NumChains"
        help = "Number of chains in system"
        arg_type = Int
        required = true

    "--CouplingFunc"
        help = "Specifies which named function to use for calculating coherent couplings"
        arg_type = String
        required = true

    "--SingleObjTimeLimit"
        help = "Time limit for single-objective optimization runs (passed to Meta.parse)"
        arg_type = String #Call Meta.parse on this string later - this allows use of multiplication in specification (e.g. --t1 24*60*60)
        required = true

    "--MaxSteps"
        help = "Maximum number of steps for BorgMOEA multi-obj algorithm (passed to Meta.parse)"
        arg_type = String
        required = true


    # Optional args

    "--NumEnsemble"
        help = "Specifies the number of runs to include in the single-objective random ensemble section"
        arg_type = Int
        default = 15

    "--Symm"
        help = "Specifies whether chains are optimized symmetrically"
        arg_type = Bool
        default = true

    "--dE_ratio"
        help = "Specifies energy gradient (relative to dipole-independent intra-chain coupling)"
        arg_type = Float64
        default = 1.0

    "--WithEs"
        help = "Sepcifies whether site energies are included in optimization"
        arg_type = Bool
        default = true

    "--WithSep"
        help = "Specified whether inter-chain separation is included in optimization"
        arg_type = Bool
        default = false

    "--ChainSep"
        help = "Initial value for inter-chain spacing"
        arg_type = Float64
        default = 1.0

    "--WithDipoles"
        help = "Specifies whether dipole orientations are included in optimization"
        arg_type = Bool
        default = false

    "--DipoleVarSites"
        help = "Specifies the sites for which dipole orientations should be optimized (passed to Meta.parse)"
        arg_type = String
        default = "nothing"

    "--EnvParams"
        help = "Specify changes to default kwargs for start model (e.g. env rates)"
        arg_type = String
        default = "nothing"

    "--ObjFunc"
        help = "specifies the objective function to be used for multi-objective optim (currently implement options: current_and_QFIM_trace, current_and_log_QFIM_volume)"
        arg_type = String
        default = "current_and_QFIM_trace"

end

cmd_args = (; parse_args(s, as_symbols=true)...,) #Convert arg dict to NamedTuple

using BiasedNetworkModels #Import after parse_args so that option errors are caught quickly

#Convert pass some parameter strings to Meta.parse
cmd_args = (; 
    cmd_args..., 
    SingleObjTimeLimit = eval(Meta.parse(cmd_args.SingleObjTimeLimit)), 
    MaxSteps = eval(Meta.parse(cmd_args.MaxSteps)),
    DipoleVarSites = eval(Meta.parse(cmd_args.DipoleVarSites)),
    EnvParams = cmd_args.EnvParams == "nothing" ? (;) : eval(Meta.parse(replace(cmd_args.EnvParams, "gamma"=>"γ"))), #Use empty NamedTuple, and replace gamma with γ since idk how to use unicode in command line args
    ObjFunc = eval(Meta.parse(cmd_args.ObjFunc)),
)

typeof(cmd_args.EnvParams) <: NamedTuple || error("You forgot to include at least 1 comma in EnvParams arg to ensure it is parsed as a NamedTuple")


#Work out which site energies should be varied during optimization based on geometry spec (usually want fixed input and output site energies)
VarSites = nothing
if (cmd_args.Geom == :sheet) || (cmd_args.Geom == :prism)
    VarSites = vcat([range((i-1)*cmd_args.ChainLength+2, i*cmd_args.ChainLength-1, step=1) for i in 1:cmd_args.NumChains]...) #For chain geom
elseif cmd_args.Geom == :ring1
    VarSites = [2:(cmd_args.ChainLength÷2)..., cmd_args.ChainLength÷2+2:cmd_args.ChainLength...] #For rings with ONE inj/ext sites
elseif cmd_args.Geom == :ring2
    VarSites = [2:(cmd_args.ChainLength÷2-1)..., cmd_args.ChainLength÷2+2:cmd_args.ChainLength-1...] #For rings with TWO inj/ext sites
else
    error("Unknown geometry spec")
end

cmd_args = (; cmd_args..., EnergyVarSites = VarSites) #Add sites for which energies should be optimized


#Check if DipoleVarSites was unset and give it a default value if so
if cmd_args.WithDipoles && cmd_args.DipoleVarSites === nothing
    cmd_args = (; cmd_args..., DipoleVarSites = 1:cmd_args.ChainLength*cmd_args.NumChains) #Set as all sites by default
end


#Parse coherent coupling specification
if cmd_args.CouplingFunc == "full_dipole_coupling"
    cmd_args = (; cmd_args..., CouplingFunc=BiasedNetworkModels.full_dipole_coupling)
elseif cmd_args.CouplingFunc == "distance_only_coupling"
    cmd_args = (; cmd_args..., CouplingFunc=BiasedNetworkModels.distance_only_coupling)
else
    error("Unrecognized CouplingFunc specification. \nValid options are: distance_only_coupling, full_dipole_coupling")
end


#Start main optimization procedure
run_params = create_run_params(; cmd_args...)
run_multi_obj_opt(run_params; obj_func = run_params.ObjFunc);
