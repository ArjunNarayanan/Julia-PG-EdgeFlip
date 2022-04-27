include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
using PyPlot
using Random

function evaluate_model(policy, env; num_trajectories = 500)
    ret = returns_versus_nflips(policy, env, num_trajectories)
    return ret
end


TS = MCTS
PV = PolicyAndValueNetwork

l2_coeff = 1e-5
discount = 1.0
batch_size = 50
memory_size = 500
num_epochs = 200

# env = EdgeFlip.OrderedGameEnv(1,10,maxflips=10)
# policy = PV.PVNet(3,16)
# optimizer = ADAM(2e-3)

# using BSON
# BSON.@save "results/models/MCTS/3L.bson" policy
# BSON.@load "results/models/MCTS/3L.bson" policy


# data = TS.BatchData(TS.initialize_state_data(env))
# TS.collect_batch_data!(data, env, policy, Cpuct, discount, maxtime, temperature, memory_size)
# target_probs, target_vals = TS.batch_target(data, discount)
# mean_vals = mean(target_vals)

# TS.train!(policy, env, optimizer, data, discount, batch_size, l2_coeff, num_epochs, evaluate_model)
# evaluate_model(policy, env)

Cpuct = 0.1
temperature = 1
maxtime = 0.1

TS.reset!(env)
p, v = TS.action_probabilities_and_value(policy, TS.state(env))
root = TS.Node(p,v,TS.is_terminal(env))
TS.search!(root, env, policy, Cpuct, discount, maxtime)
ap = TS.mcts_action_probabilities(root.visit_count, 72, temperature)

tree_returns_versus_nflips(policy, Cpuct, discount, maxtime, temperature, 1, 10, 10)