using EdgeFlip
include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
# using PyPlot
using Random

TS = MCTS
PV = PolicyAndValueNetwork
EF = EdgeFlip

function evaluate_model(policy, env; num_trajectories = 500)
    ret = returns_versus_nflips(policy, env, num_trajectories)
    return ret
end

function keep_new_policy(old_policy, new_policy, env; num_samples = 500, threshold = 0.6)
    returns = zeros(2,num_samples)
    for sample in 1:num_samples
        TS.reset!(env)
        returns[1,sample] = single_trajectory_normalized_return(env, old_policy)
        EF.reset!(env, fixed = true)
        returns[2,sample] = single_trajectory_normalized_return(env, new_policy)
    end

    win_rate = count(returns[2,:] .> returns[1,:])/num_samples

    return win_rate/num_samples
end



env = EdgeFlip.OrderedGameEnv(1,10,maxflips=10)
policy = PV.PVNet(8,64)

# using BSON
# BSON.@save "results/models/MCTS/8L.bson" policy
BSON.@load "results/models/MCTS/8L.bson" policy


exploration_factor = 1.5
maxiter = 500
temperature = 1
discount = 1.0
tree_settings = TS.TreeSettings(exploration_factor, maxiter, temperature, discount)

l2_coeff = 1e-3
memory_size = 500
num_epochs = 200
batch_size = 50
num_iter = 100
iter_settings = TS.PolicyIterationSettings(discount, l2_coeff, memory_size, batch_size, num_epochs, num_iter)

data = TS.BatchData(TS.initialize_state_data(env))
TS.collect_batch_data!(data, env, policy, tree_settings, memory_size)
target_probs, target_vals = TS.batch_target(data, discount);
mean_vals = mean(target_vals)

# optimizer = ADAM(1e-2)
# TS.train!(policy, env, optimizer, data, discount, batch_size, l2_coeff, num_epochs, evaluate_model)

# evaluate_model(policy, env)
win_rate = keep_new_policy(policy, policy, env)

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
