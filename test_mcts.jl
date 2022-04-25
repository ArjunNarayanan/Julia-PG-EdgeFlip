include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
# using MeshPlotter

function evaluate_model(policy, env; num_trajectories = 500)
    ret = returns_versus_nflips(policy, env, num_trajectories)
    return ret
end


TS = MCTS
PV = PolicyAndValueNetwork
# MP = MeshPlotter

Cpuct = 10
temperature = 50
l2_coeff = 1e-3
discount = 1.0
maxtime = 1e-2
batch_size = 50
num_epochs = 50


nref = 1
nflips = 10
env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = nflips)
policy = PV.PVNet(3, 16)


optimizer = ADAM(1e-2)
TS.train!(policy, env, optimizer, Cpuct, discount, maxtime, temperature, batch_size, l2_coeff, num_epochs, evaluate_model)