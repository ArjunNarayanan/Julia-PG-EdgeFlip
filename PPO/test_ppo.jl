using EdgeFlip
include("PPO_utilities.jl")

function evaluator(policy, env; num_trajectories = 100)
    ret, dev = average_normalized_returns(env, policy, num_trajectories)
    return ret, dev
end


episodes_per_iteration = 1000
discount = 1.0
epsilon = 0.1
batch_size = 200
num_epochs = 5
num_iter = 50

env = EF.OrderedGameEnv(2, 20, maxflips = 20)
policy = Policy.DirectPolicy(16, 16, 3)
# value = Value.ValueNL(3, 16)

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

