
using Pkg; Pkg.activate("test-env", shared=true)
using BiasedNetworkModels

### General constructor tests
dE = 1.0
chain_length = 4
num_chains = 2

#Sheet Hamiltonians
@time H_sheet1 = BiasedSheetHamiltonian(chain_length, num_chains, dE)
@time H_sheet1 = BiasedSheetHamiltonian(chain_length, num_chains, dE; interchain_coupling=0.5)
@time H_sheet2 = BiasedSheetHamiltonian(chain_length, num_chains, dE; dipole_orientations=[rand(3) for i in 1:chain_length*num_chains])

# #Prism Hamiltonians
# H_prism1 = BiasedPrismHamiltonian(dE, chain_length, num_chains, dipole_angles=:parallel)
# H_prism2 = BiasedPrismHamiltonian(dE, chain_length, num_chains, dipole_angles=:antiparallel)
# H_prism3 = BiasedPrismHamiltonian(dE, chain_length, num_chains, dipole_angles=:random)