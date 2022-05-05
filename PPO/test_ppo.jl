using EdgeFlip
include("PPO_utilities.jl")


discount = 1.0
epsilon = 0.1
env = EF.OrderedGameEnv(1, 10, maxflips = 10)
policy = Policy.PolicyNL(3, 32)
optimizer = ADAM(0.01)

data = PPO.BatchData(PPO.initialize_state_data(env))
PPO.collect_batch_data!(data, env, policy, 100)
PPO.step_batch!(policy, optimizer, data, discount, epsilon)
