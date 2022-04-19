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
batch_size = 50

data = TS.BatchData(StateData(EdgeFlip.edge_connectivity(env)))
TS.collect_batch_data!(data, env, policy, Cpuct, discount, maxtime, temperature, batch_size)

ets, econn, epairs, nflips = batch_data(data.state_data)
policy_probs, policy_vals = PV.eval_batch(policy, ets, econn, epairs, nflips)
target_probs, target_vals = TS.batch_target(data, discount)

# ets, econn, epairs = batch_data(s)
# h = PV.eval_batch(policy.emodels[1], ets, econn, epairs)