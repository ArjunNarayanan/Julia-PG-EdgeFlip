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


env = EdgeFlip.OrderedGameEnv(1,10,maxflips=10)
# policy = PV.PVNet(3,16)

using BSON
# BSON.@save "results/models/MCTS/3L.bson" policy
BSON.@load "results/models/MCTS/3L.bson" policy


root = TS.Node(env, policy)

# probability_weight = 15

# l2_coeff = 1e-5
# memory_size = 500
# num_epochs = 200
batch_size = 50
discount = 1.0

exploration_factor = 1
maxiter = 500
temperature = 0.05
tree_settings = TS.TreeSettings(probability_weight, exploration_factor, maxiter, temperature, discount)
data = TS.BatchData(TS.initialize_state_data(env))
TS.collect_batch_data!(data, env, policy, tree_settings, batch_size)
target_probs, target_vals = TS.batch_target(data, discount)
mean_vals = mean(target_vals)

# optimizer = ADAM(5e-3)
# TS.train!(policy, env, optimizer, data, discount, batch_size, l2_coeff, num_epochs, evaluate_model)

# evaluate_model(policy, env)


# ret = average_normalized_tree_returns(
#     env,
#     policy,
#     tree_exploration_factor,
#     probability_weight,
#     discount,
#     maxiter,
#     temperature,
#     10,
# )

# tree_returns_versus_nflips(policy, Cpuct, discount, maxtime, temperature, 1, 10, 10)
