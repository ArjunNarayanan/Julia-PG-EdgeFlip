using Flux 
using EdgeFlip
include("PPO_direct_policy.jl")


env = EdgeFlip.OrderedGameEnv(1, 10, maxflips = 10)
x = env.edge_template_score[[1],:]
p = env.edge_pairs
p[p .== 0] .= length(p)+1

policy = Policy.DirectPolicy(8, 16, 2)

Policy.eval_single(policy, x, p)