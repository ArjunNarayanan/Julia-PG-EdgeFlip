using EdgeFlip
include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
# using PyPlot

TS = MCTS
PV = PolicyAndValueNetwork
EF = EdgeFlip

function old_vs_new_policy(old_policy, new_policy, env, num_samples)
    returns = zeros(2, num_samples)
    for sample = 1:num_samples
        TS.reset!(env)
        returns[1, sample] = single_trajectory_normalized_return(env, old_policy)
        EF.reset!(env, fixed = true)
        returns[2, sample] = single_trajectory_normalized_return(env, new_policy)
    end
    return returns
end

env = EdgeFlip.OrderedGameEnv(1, 10, maxflips = 10)
# policy = PV.PVNet(8, 64)

# using BSON
# BSON.@save "results/models/8L-MCTS/policy-1.bson" policy
# BSON.@load "results/models/8L-MCTS/policy-1.bson" policy


exploration_factor = 1.5
maxiter = 500
temperature = 1
discount = 1.0
tree_settings = TS.TreeSettings(exploration_factor, maxiter, temperature, discount)

l2_coeff = 1e-3
memory_size = 500
num_epochs = 200
batch_size = 50
num_iter = 10
threshold = 0.6
num_samples = 500

iter_settings = TS.PolicyIterationSettings(
    discount,
    l2_coeff,
    memory_size,
    batch_size,
    num_epochs,
    num_iter,
    threshold,
    num_samples,
)

learning_rate = 1e-2
decay = 0.7
decay_step = 500
clip = 5e-5
optimizer =
    Flux.Optimiser(ExpDecay(learning_rate, decay, decay_step, clip), ADAM(learning_rate))

best_policy = TS.iterate!(
    policy,
    env,
    optimizer,
    old_vs_new_policy,
    tree_settings,
    iter_settings,
    foldername = "results/models/8L-MCTS/",
)

ret, dev = average_normalized_returns(env, best_policy, 500)