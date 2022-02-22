using EdgeFlip
include("two_edge_testbed.jl")

env = two_edge_game_env()

states = all_states()
rewards = all_rewards(env,states)

optimum = vec(maximum(rewards,dims=1))
exp_opt_ret = sum(optimum)/length(optimum)