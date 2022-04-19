using EdgeFlip
using Distributions: Categorical

include("MCTS_utilities.jl")
include("policy_and_value_network.jl")

TS = MCTS
PV = PolicyAndValueNetwork

nref = 1
env = EdgeFlip.OrderedGameEnv(nref, 0)
policy = PV.PVNet(3, 16)

Cpuct = 1.0
temperature = 1.0
discount = 1.0
maxtime = 1e-2
batch_size = 55

batch_data = TS.BatchData(StateData(EdgeFlip.edge_connectivity(env)))

TS.collect_batch_data!(batch_data, env, policy, Cpuct, discount, maxtime, temperature, batch_size)

s = batch_data.state_data

ets = cat(s.edge_template_score...,dims=3)
econn = s.edge_connectivity
epairs = cat(s.edge_pairs...,dims=2)