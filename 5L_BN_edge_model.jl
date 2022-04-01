using Printf
using Flux
using EdgeFlip
include("supervised_greedy_training.jl")
include("edge_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = EdgePolicyGradient
SV = Supervised

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

function PG.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    return ets, econn, epairs
end

SV.state(env::EdgeFlip.OrderedGameEnv) = PG.state(env)
SV.reset!(env::EdgeFlip.OrderedGameEnv) = PG.reset!(env)

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

function PG.reset!(
    env::EdgeFlip.OrderedGameEnv;
    nflips = rand(1:EdgeFlip.number_of_actions(env)),
)
    maxflips = ceil(Int, 1.2nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end

struct EdgeModel
    model::Any
    bvals::Any
    batchnorm::Any
    function EdgeModel(in_channels, out_channels)
        model = Dense(3in_channels, out_channels)
        bvals = Flux.glorot_uniform(in_channels)
        batchnorm = BatchNorm(out_channels)
        new(model, bvals, batchnorm)
    end
end

Flux.@functor EdgeModel

function eval_single(em::EdgeModel, ep, econn, epairs)
    nf, na = size(ep)

    ep = cat(ep, em.bvals, dims = 2)
    ep = reshape(ep[:, econn], 3nf, na)
    ep = em.model(ep)

    ep2 = ep[:, epairs]
    ep = 0.5 * (ep + ep2)

    ep = em.batchnorm(ep)

    return ep
end

function eval_batch(em::EdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = cat(ep, repeat(em.bvals, inner = (1, 1, nb)), dims = 2)
    ep = reshape(ep[:, econn, :], 3nf, na, nb)
    ep = em.model(ep)

    ep2 = cat([ep[:, epairs[:, b], b] for b = 1:nb]..., dims = 3)
    ep = 0.5 * (ep + ep2)

    nf, na, nb = size(ep)
    ep = reshape(ep, nf, :)
    ep = em.batchnorm(ep)
    ep = reshape(ep, nf, na, nb)

    return ep
end

struct Policy5L
    emodel1::Any
    emodel2::Any
    emodel3::Any
    emodel4::Any
    emodel5::Any
    lmodel::Any
    function Policy5L(hidden_channels)
        emodel1 = EdgeModel(4, hidden_channels)
        emodel2 = EdgeModel(hidden_channels, hidden_channels)
        emodel3 = EdgeModel(hidden_channels, hidden_channels)
        emodel4 = EdgeModel(hidden_channels, hidden_channels)
        emodel5 = EdgeModel(hidden_channels, hidden_channels)
        lmodel = Dense(hidden_channels, 1)
        new(emodel1, emodel2, emodel3, emodel4, emodel5, lmodel)
    end
end

Flux.@functor Policy5L

function PG.eval_single(p::Policy5L, ep, econn, epairs)
    x = eval_single(p.emodel1, ep, econn, epairs)
    x = relu.(x)

    y = eval_single(p.emodel2, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_single(p.emodel3, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_single(p.emodel4, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_single(p.emodel5, x, econn, epairs)
    y = x + y
    x = relu.(y)

    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::Policy5L, ep, econn, epairs)
    x = eval_batch(p.emodel1, ep, econn, epairs)
    x = relu.(x)

    y = eval_batch(p.emodel2, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_batch(p.emodel3, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_batch(p.emodel4, x, econn, epairs)
    y = x + y
    x = relu.(y)

    y = eval_batch(p.emodel5, x, econn, epairs)
    y = x + y
    x = relu.(y)

    logits = p.lmodel(x)
    return logits
end

nref = 1

env = EdgeFlip.OrderedGameEnv(nref, 0)
num_actions = EdgeFlip.number_of_actions(env)
policy = Policy5L(8)

# PG.reset!(env)
# ep, econn, epairs = PG.state(env)
# l = PG.eval_single(policy, ep, econn, epairs)
# bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, 10, 1.0)
# l = PG.eval_batch(policy, bs[1], bs[2], bs[3])

# num_trajectories = 500
nflip_range = 1:5:42
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# normalized_nflips = nflip_range ./ num_actions


num_epochs = 5000
batch_size = 100
discount = 1.0

# optimizer = Flux.Optimiser(ExpDecay(0.1, 0.7, 500, 1e-4), ADAM(0.01))
optimizer = Flux.Optimiser(ExpDecay(0.1, 0.7, 500, 1e-6), ADAM(1e-4))
PG.run_training_loop(env, policy, optimizer, batch_size, discount, num_epochs)

num_trajectories = 500
nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1])

# filename = "results/new-edge-model/5L-res-performance.png"
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1], filename = filename)