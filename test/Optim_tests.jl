

using BiasedNetworkModels, OQSmodels
using BiasedNetworkModels: NetworkOptim

RP = NetworkOptim.create_run_params(
    Geom=:sheet, ChainLength=4, NumChains=2, 
    EnergyVarSites=[2, 3, 6, 7], 
    NumEnsemble=5,
    SingleObjTimeLimit = 10,
    SingleObjMaxIters = 10^5,
    # WithSep=true,
    # WithDipoles=true, DipoleVarSites=1:3,
)


m = NetworkOptim.init_model(RP)
x0 = NetworkOptim.get_x(m, RP)

NetworkOptim.update_x(m, zeros(2), RP)

NetworkOptim.perturb_x(m, RP)

NetworkOptim.FIM_params(m, RP)


sol_list = NetworkOptim.run_ensemble_opt(m, RP);