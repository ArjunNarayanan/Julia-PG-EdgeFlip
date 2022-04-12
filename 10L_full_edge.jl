using Printf
using Flux
using EdgeFlip
include("full_edge_policy_gradient.jl")
include("full_edge_NL_policy.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = FullEdgePolicyGradient

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.0)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.FullEdgeGameEnv(nref, nflips, maxflips = maxflips)
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

function PG.is_terminated(env::EdgeFlip.FullEdgeGameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.FullEdgeGameEnv)
    return EdgeFlip.reward(env)
end

function PG.reset!(env::EdgeFlip.FullEdgeGameEnv; nflips = 11, maxflipfactor = 1.0)
    maxflips = ceil(Int, maxflipfactor*nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.FullEdgeGameEnv)
    return EdgeFlip.score(env)
end

function evaluate_model(policy; num_trajectories = 500)
    nref = 1
    nflip_range = [11]
    ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
    return ret
end


nref = 1
env = EdgeFlip.FullEdgeGameEnv(nref, 0)
policy = FullEdgePolicyNL(10,16)

# PG.reset!(env)
# ep, econn = PG.state(env)
# l = PG.eval_single(policy, ep, econn)

# bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, 10, 1.0)
# ets, econn = bs

# l = PG.eval_batch(policy, ets, econn)
# loss = PG.policy_gradient_loss(ets, econn, policy, ba, bw)

# ep = reshape(ets, 4, :)
# l2 = PG.eval_single(policy, ep, econn)
# l1 = eval_batch(policy.emodels[1], ets, econn)

# econn2 = copy(econn)
# econn2[econn2 .== 73] .= 0
# econn2 = vec(econn2)

# ets2 = reshape(ets, 4, :)
# econn2[econn2 .== 0] .= size(ets2,2)+1

# l2 = eval_single(policy.emodels[1], ets2, econn2)
# l1 = reshape(l1, 16, :)

# loss = PG.policy_gradient_loss(ets, econn, policy, ba, bw)

# using BenchmarkTools

# optimizer = ADAM()
# PG.step_epoch(env, policy, optimizer, 10, 0.9)
# @btime PG.step_epoch(env, policy, optimizer, 10, 0.9)

# PG.reset!(env)
# ret = PG.average_normalized_returns(env, policy, 500)

batch_size = 100
num_epochs = 10000
learning_rate = 1e-2
decay = 0.7
decay_step = 500
clip = 5e-5
discount = 0.9

optimizer =
    Flux.Optimiser(ExpDecay(learning_rate, decay, decay_step, clip), ADAM(learning_rate))
optimizer = ADAM(2e-5)

PG.train_and_save_best_models(
    env,
    policy,
    optimizer,
    batch_size,
    discount,
    num_epochs,
    evaluate_model,
    foldername = "results/models/10L-full-edge/",
    generate_plots = false
)

# PG.reset!(env)
# initial_score = PG.score(env)

# ets, econn = PG.state(env)
# probs = softmax(vec(policy(ets, econn)))
# action_index = rand(Categorical(probs))

# action = PG.idx_to_action(action_index)
# PG.step!(env, action)

# new_score = PG.score(env)

# ret = PG.single_trajectory_return(env, policy)