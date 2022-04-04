using Flux
using EdgeFlip
using BenchmarkTools
include("tri.jl")
include("edge_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = EdgePolicyGradient

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

function PG.reset!(env::EdgeFlip.OrderedGameEnv)
    EdgeFlip.reset!(env)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end

struct Policy3L
    emodel1::Any
    emodel2::Any
    emodel3::Any
    lmodel::Any
    function Policy3L()
        emodel1 = EdgeModel(4, 16)
        emodel2 = EdgeModel(16, 16)
        emodel3 = EdgeModel(16, 16)
        lmodel = Dense(16, 1)
        new(emodel1, emodel2, emodel3, lmodel)
    end
end

Flux.@functor Policy3L

function PG.eval_single(p::Policy3L, ep, econn, epairs)
    x = eval_single(p.emodel1, ep, econn, epairs)
    x = relu.(x)

    y = eval_single(p.emodel2, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_single(p.emodel3, x, econn, epairs)
    y = x + y
    x = relu.(y)

    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::Policy3L, ep, econn, epairs)
    x = eval_batch(p.emodel1, ep, econn, epairs)
    x = relu.(x)

    y = eval_batch(p.emodel2, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_batch(p.emodel3, x, econn, epairs)
    y = x + y
    x = relu.(y)

    logits = p.lmodel(x)
    return logits
end

function make_env(element_size; maxflipfactor = 2)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    num_edges = EdgeFlip.number_of_edges(mesh)
    d0 = fill(6, num_nodes)
    d0[mesh.bnd_nodes] .= 4
    maxflips = ceil(Int, maxflipfactor * num_edges)

    env = EdgeFlip.OrderedGameEnv(mesh, 0, d0 = d0, maxflips = maxflips)

    return env
end

function fake_loss(ets, econn, epairs, policy)
    logits = PG.eval_batch(policy, ets, econn, epairs)
    # logits = eval_batch(policy.emodel1, ets, econn, epairs)
    return sum(logits)
end

function eval_gradient(states, policy, actions, weights)
    theta = Flux.params(policy)
    grads = Flux.gradient(theta) do
        # loss = PG.policy_gradient_loss(states[1], states[2], states[3], policy, actions, weights)
        loss = fake_loss(states[1], states[2], states[3], policy)
    end
    return grads
end

element_size = 0.4
maxflipfactor = 2

env = make_env(element_size)
policy = Policy3L()

PG.reset!(env)
bs, ba, bw = PG.collect_batch_trajectories(env, policy, 10, 1.0)
# @btime bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, 10, 1.0)

eval_gradient(bs, policy, ba, bw)
@btime eval_gradient(bs, policy, ba, bw)
# @btime PG.policy_gradient_loss(bs[1], bs[2], bs[3], policy, ba, bw)
# grads = eval_gradient(bs, policy, ba, bw)
# @btime Flux.update!(optimizer, Flux.params(policy), grads)

# l = PG.eval_batch(policy, bs[1], bs[2], bs[3])

# num_trajectories = 500
# nflip_range = 1:5:42
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# normalized_nflips = nflip_range ./ num_actions

# num_epochs = 1000
# batch_size = 200
# discount = 1.0
# learning_rate = 1e-2

# optimizer = ADAM(learning_rate)
# epochs, ret = PG.run_training_loop(env, policy, optimizer, batch_size, discount, num_epochs)

# num_trajectories = 500
# nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1])

# filename = "results/new-edge-model/3L-res-performance.png"
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1], filename = filename)