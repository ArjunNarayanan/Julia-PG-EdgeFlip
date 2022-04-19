using EdgeFlip
using Distributions: Categorical

include("MCTS_utilities.jl")
include("policy_and_value_network.jl")

TS = MCTS
PV = PolicyAndValueNetwork

nref = 0
env = EdgeFlip.OrderedGameEnv(nref, 0)
policy = PV.PVNet(3, 16)

Cpuct = 1.0
temperature = 1.0
discount = 1.0
maxtime = 0.1

TS.reset!(env)
batch_data = TS.BatchData(StateData())

TS.collect_sample_trajectory!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)

# TS.reverse_step!(env,c.action)
# TS.step!(env,c.action)
# r = TS.reward(env)