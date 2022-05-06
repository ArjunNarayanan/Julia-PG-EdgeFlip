using Printf
using Flux
using EdgeFlip
include("../edge_policy_gradient.jl")
include("simple_edge_policy.jl")

PG = EdgePolicyGradient
EF = EdgeFlip
SP = SimplePolicy

function PG.state(env)

    x = cat(EF.desired_degree(env)', EF.degree(env)', dims = 1)
    epairs = copy(EF.edge_pairs(env))

    return x, epairs
end

function PG.is_terminated(env)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env)
    return EF.reward(env)
end

function PG.reset!(env)
    EF.reset!(env)
end

function PG.step!(env, action)
    triangle, vertex = action
    EdgeFlip.step!(env, triangle, vertex)
end


env = EdgeFlip.SimpleGameEnv(1, 10)
num_actions = EdgeFlip.number_of_actions(env)
policy = SP.SPolicy(1, 3, 32)

s = PG.state(env)