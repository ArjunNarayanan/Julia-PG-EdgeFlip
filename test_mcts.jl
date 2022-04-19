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
l2_coeff = 1e-2
discount = 1.0
maxtime = 1e-2
batch_size = 50

optimizer = ADAM(0.01)

TS.step_epoch!(policy, env, optimizer, Cpuct, discount, maxtime, temperature, batch_size, l2_coeff)

# data = TS.BatchData(StateData(env))
# TS.collect_batch_data!(data, env, policy, Cpuct, discount, maxtime, temperature, batch_size)

# state = batch_state(data.state_data)
# target_probs, target_vals = TS.batch_target(data, discount)

# l = TS.loss(policy, state, target_probs, target_vals, l2_coeff)

# ets, econn, epairs = batch_data(s)
# h = PV.eval_batch(policy.emodels[1], ets, econn, epairs)