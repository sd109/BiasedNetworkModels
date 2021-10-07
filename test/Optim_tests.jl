
using BiasedNetworkModels, OQSmodels, BlackBoxOptim
using BiasedNetworkModels: NetworkOptim

N_sites = 12
N_chains = 2
RP = NetworkOptim.create_run_params(
    SaveName = "12site-2parallel-withEs",
    Geom=:sheet, ChainLength=N_sites, NumChains=N_chains, 
    EnergyVarSites=[2:11..., 14:23...], 
    NumEnsemble=15,
    SingleObjTimeLimit = 3600,
    SingleObjMaxIters = 10^6,
    # WithSep=true, ChainSep=1.0,
    # WithDipoles=true, DipoleVarSites=1:3,
    MaxSteps=10^5,
)


m = NetworkOptim.init_model(RP)
x0 = NetworkOptim.get_x(m, RP)

NetworkOptim.update_x(m, x0, RP)
# NetworkOptim.update_x(m, [zeros(2)..., 2.0], RP)

NetworkOptim.perturb_x(m, RP)

NetworkOptim.FIM_params(m, RP)

sol_list = NetworkOptim.run_ensemble_opt(m, RP);
length(sol_list)
sol_list[end].minimum
sol_list[end].u


save_data = NetworkOptim.run_multi_obj_opt(RP);
@show length(save_data.pf)
@show fitness.(save_data.pf)