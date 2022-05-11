using EdgeFlip
include("PPO_utilities.jl")

function evaluator(policy, env; num_trajectories = 100)
    ret, dev = average_normalized_returns(env, policy, num_trajectories)
    return ret, dev
end


episodes_per_iteration = 500
discount = 1.0
epsilon = 0.2   
batch_size = 50
num_epochs = 10
num_iter = 50
env = EF.OrderedGameEnv(1, 10, maxflips = 10)

policy = Policy.DirectPolicy(8, 16, 2)
# value = Value.ValueNL(3, 16)

data = PPO.BatchData()
PPO.collect_batch_data!(data, env, policy, episodes_per_iteration)

# learning_rate = 1e-3
# decay = 0.8
# decay_step = 50
# clip = 5e-5
# optimizer =
#     Flux.Optimiser(ExpDecay(learning_rate, decay, decay_step, clip), ADAM(learning_rate))
optimizer = ADAM(1e-3)

PPO.ppo_iterate!(
    policy,
    env,
    optimizer,
    episodes_per_iteration,
    discount,
    epsilon,
    batch_size,
    num_epochs,
    num_iter,
    evaluator,
)
