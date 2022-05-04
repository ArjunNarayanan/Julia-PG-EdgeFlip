using EdgeFlip
using MeshPlotter
include("greedy_policy.jl")
include("circlemesh.jl")

GP = GreedyPolicy

element_size = 0.2
env = circle_env(element_size, maxflipfactor = 2)

ret = GP.normalized_returns(env, 100)