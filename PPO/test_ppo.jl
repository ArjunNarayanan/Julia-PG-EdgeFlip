using EdgeFlip
include("PPO_utilities.jl")


discount = 1.0
epsilon = 0.1
memory_size = 100
env = EF.OrderedGameEnv(1, 10, maxflips = 10)
policy = Policy.PolicyNL(2, 10)

PPO.reset!(env)
data = PPO.BatchData(PPO.initialize_state_data(env))
PPO.collect_batch_data!(data, env, policy, memory_size)
PPO.step_batch!(policy, optimizer, data, discount, epsilon)


optimizer = ADAM(0.01)

PPO.reset!(env)
ret = single_trajectory_normalized_return(env, policy)