using EdgeFlip
using Printf
include("greedy_policy.jl")

function GreedyPolicy.reset!(env::EdgeFlip.GameEnv; vertex_score = rand(-3:3,5))
    d0 = env.mesh.d - vertex_score
    EdgeFlip.reset!(env, d0 = d0)
end


function two_edge_game_env(; vertex_score = rand(-3:3, 5))
    t = [
        1 2 4
        1 3 2
        1 5 3
    ]
    p = rand(5, 2)
    mesh = EdgeFlip.Mesh(p, t)

    env = EdgeFlip.GameEnv(mesh, 0, d0 = mesh.d - vertex_score, maxflips = 2)
    return env
end

env = two_edge_game_env()
ret = GreedyPolicy.average_return(env, 100000)