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


# using BSON
# BSON.@save "results/models/MCTS/3L.bson" policy
BSON.@load "results/models/MCTS/3L.bson" policy


data = TS.BatchData(TS.initialize_state_data(env))
TS.collect_batch_data!(data, env, policy, tree_exploration_factor, probability_weight, discount, maxiter, temperature, memory_size)
target_probs, target_vals = TS.batch_target(data, discount)
mean_vals = mean(target_vals)

optimizer = ADAM(5e-3)
TS.train!(policy, env, optimizer, data, discount, batch_size, l2_coeff, num_epochs, evaluate_model)
evaluate_model(policy, env)


tree_exploration_factor = 0.25
probability_weight = 40
maxiter = 500
temperature = 1e0

ret = average_normalized_tree_returns(
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
    10,
)

# tree_returns_versus_nflips(policy, Cpuct, discount, maxtime, temperature, 1, 10, 10)
