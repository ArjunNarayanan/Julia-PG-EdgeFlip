using Printf
using Flux
using EdgeFlip
include("full_edge_policy_gradient.jl")
include("full_edge_NL_policy.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = FullEdgePolicyGradient

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = maxflips)
    avg = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" env.num_initial_flips env.maxflips avg
    return avg
end

function returns_versus_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" env.num_initial_flips env.maxflips avg
    return avg
end

function PG.state(env::EdgeFlip.FullEdgeGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    nf, na = size(ets)

    econn = copy(vec(EdgeFlip.edge_connectivity(env)))

    idx = findall(econn .== 0)
    econn[idx] .= na + 1

    return ets, econn
end

function PG.step!(env::EdgeFlip.FullEdgeGameEnv, action)
    triangle, vertex = action
    EdgeFlip.step!(env, triangle, vertex)
end

function PG.is_terminated(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.reward(env)
end

# function PG.reset!(
#     env::EdgeFlip.OrderedGameEnv;
#     nflips = rand(1:EdgeFlip.number_of_actions(env)),
# )
#     maxflips = ceil(Int, 1.2nflips)
#     EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
# end

function PG.reset!(env::EdgeFlip.OrderedGameEnv; nflips = 11, maxflipfactor = 1.0)
    maxflips = ceil(Int, maxflipfactor*nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end

function evaluate_model(policy; num_trajectories = 500)
    nref = 1
    # nflip_range = 1:5:42
    nflip_range = [11]
    ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
    return ret
end



nref = 1
env = EdgeFlip.FullEdgeGameEnv(nref, 0)
policy = FullEdgePolicyNL(3,16)

ets, econn = PG.state(env)

ep = PG.eval_single(policy, ets, econn)