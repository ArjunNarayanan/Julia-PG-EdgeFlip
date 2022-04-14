using Printf
using Flux
using EdgeFlip
include("edge_policy_gradient.jl")
include("NL_policy.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = EdgePolicyGradient

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.0)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = maxflips)
    avg = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" env.num_initial_flips env.maxflips avg
    return avg
end

function returns_versus_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.0)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" env.num_initial_flips env.maxflips avg
    return avg
end

function PG.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    return ets, econn, epairs
end

function PG.step!(env::EdgeFlip.OrderedGameEnv, action)
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

env = EdgeFlip.OrderedGameEnv(nref, 0)
num_actions = EdgeFlip.number_of_actions(env)
policy = PolicyNL(10,16)

# PG.reset!(env)
# ep, econn, epairs = PG.state(env)
# l = PG.eval_single(policy, ep, econn, epairs)
# bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, 10, 1.0)
# l = PG.eval_batch(policy, bs[1], bs[2], bs[3])

# num_trajectories = 500
# nflip_range = 1:5:42
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# normalized_nflips = nflip_range ./ num_actions


batch_size = 100
num_epochs = 10000
learning_rate = 1e-2
decay = 0.7
decay_step = 500
clip = 5e-5
discount = 0.8

optimizer =
    Flux.Optimiser(ExpDecay(learning_rate, decay, decay_step, clip), ADAM(learning_rate))
# # optimizer = ADAM(5e-6)

PG.train_and_save_best_models(
    env,
    policy,
    optimizer,
    batch_size,
    discount,
    num_epochs,
    evaluate_model,
    foldername = "results/models/10L-model/"
)


nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories, maxstepfactor = 1.2) for nf in nflip_range]
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1])
