using Printf
using Flux
using EdgeFlip
# include("supervised_greedy_training.jl")
include("edge_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

PG = EdgePolicyGradient

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

    boundary_index = size(ets, 2) + 1
    econn[econn .== 0] .= boundary_index
    
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

function PG.reset!(env::EdgeFlip.OrderedGameEnv; nflips = rand(1:42))
    maxflips = ceil(Int, 1.2nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end

struct EdgeModel
    model
    bvals
    function EdgeModel(in_channels, out_channels)
        model = Dense(3in_channels, out_channels)
        bvals = Flux.glorot_uniform(in_channels)
        new(model, bvals)
    end
end

Flux.@functor EdgeModel

function eval_single(em::EdgeModel, ep, econn, epairs)
    nf, na = size(ep)

    ep = cat(ep, em.bvals, dims = 2)
    ep = reshape(ep[:, econn], 3nf, na)
    ep = em.model(ep)

    ep2 = ep[:, epairs]
    ep = 0.5*(ep + ep2)
    
    return ep
end

function eval_batch(em::EdgeModel, ep, econn, epairs)
    nf, na, nb = size(ep)

    ep = cat(ep, repeat(em.bvals, inner = (1,1,nb)), dims = 2)
    ep = [reshape(ep[:, econn[:, b], b], 3nf, na) for b in 1:nb]
    ep = cat(ep..., dims = 3)
    ep = em.model(ep)

    ep2 = cat([ep[:, epairs[:, b], b] for b in 1:nb]..., dims = 3)
    ep = 0.5*(ep + ep2)
    
    return ep
end

struct Policy1L
    emodel::Any
    lmodel
    function Policy1L()
        emodel = EdgeModel(4,8)
        lmodel = Dense(8,1)
        new(emodel, lmodel)
    end
end

Flux.@functor Policy1L

function PG.eval_single(p::Policy1L, ets, econn, epairs)
    x = eval_single(p.emodel, ets, econn, epairs)
    x = relu.(x)

    logits = p.lmodel(x)
    return logits
end

function PG.eval_batch(p::Policy1L, ets, econn, epairs)
    x = eval_batch(p.emodel, ets, econn, epairs)
    x = relu.(x)

    logits = p.lmodel(x)
    return logits
end

nref = 1

env = EdgeFlip.OrderedGameEnv(nref, 0)
num_actions = EdgeFlip.number_of_actions(env)
policy = Policy1L()

# num_trajectories = 500
# nflip_range = 1:5:42
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# normalized_nflips = nflip_range ./ num_actions

num_epochs = 1000
batch_size = 100
discount = 1.0
learning_rate = 2e-3

PG.run_training_loop(env, policy, batch_size, discount, num_epochs, learning_rate)
nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1])
# filename = "results/new-edge-model/1L-no-avg.png"
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1], filename = filename)

# using BSON: @save
# @save "results/models/new-edge-model/1L.bson" policy