
using Pkg; Pkg.activate("./BiasedNetworkModels")
using BiasedNetworkModels, OQSmodels, BlackBoxOptim

N_sites = 12
N_chains = 2
RP = create_run_params(
    SaveName = "test",
    Geom=:sheet, ChainLength=N_sites, NumChains=N_chains, 
    CouplingFunc = BiasedNetworkModels.full_dipole_coupling,
    EnergyVarSites=[2:11..., 14:23...], 
    NumEnsemble=2,
    SingleObjTimeLimit = 10,
    SingleObjMaxIters = 10^6,
    WithSep=true, ChainSep=1.0,
    WithDipoles=true, DipoleVarSites=1:N_sites*N_chains,
    MaxSteps=10^2,
)


m = create_init_model(RP)
@show ss_current(m)
x0 = BiasedNetworkModels.get_x(m, RP)

update_x(m, x0, RP)
# NetworkOptim.update_x(m, [zeros(2)..., 2.0], RP)

# NetworkOptim.perturb_x(m, RP)

# NetworkOptim.FIM_params(m, RP)

# sol_list = NetworkOptim.run_ensemble_opt(m, RP);
# length(sol_list)
# sol_list[end].minimum
# sol_list[end].u


save_data = run_multi_obj_opt(RP);
@show length(save_data.pf)
@show fitness.(save_data.pf)