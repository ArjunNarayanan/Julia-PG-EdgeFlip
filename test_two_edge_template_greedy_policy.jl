using EdgeFlip
using Printf
include("greedy_policy.jl")

function GreedyPolicy.step!(env::EdgeFlip.GameEnv, action)
    num_actions = EdgeFlip.number_of_actions(env)
    if 0 < action <= num_actions
        EdgeFlip.step!(env, action)
    else
        EdgeFlip.step!(env)
    end
end

function GreedyPolicy.is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function GreedyPolicy.reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function GreedyPolicy.reset!(env::EdgeFlip.GameEnv; vertex_score = rand(-3:3,5))
    d0 = env.mesh.d - vertex_score
    EdgeFlip.reset!(env, d0 = d0)
end

function GreedyPolicy.score(env::EdgeFlip.GameEnv)
    return EdgeFlip.score(env)
end


function two_edge_game_env(; vertex_score = rand(-3:3, 5))
    t = [
        1 2 5
        2 4 5
        2 3 4
    ]
    p = rand(5, 2)
    mesh = EdgeFlip.Mesh(p, t)

    env = EdgeFlip.GameEnv(mesh, 0, d0 = mesh.d - vertex_score, maxflips = 0)
    return env
end


env = two_edge_game_env()
avg, dev = GreedyPolicy.mean_and_std_returns(env,2,100000)

# GreedyPolicy.reset!(env)
