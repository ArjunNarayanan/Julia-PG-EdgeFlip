using Printf
using Flux
using EdgeFlip
# include("supervised_greedy_training.jl")
include("tri.jl")
include("edge_policy_gradient.jl")
include("edge_model.jl")
include("greedy_policy.jl")
# include("plot.jl")

PG = EdgePolicyGradient
GP = GreedyPolicy

function returns_versus_maxflips(policy, element_size, maxflips, num_trajectories)
    env = make_env(element_size, maxflips)
    ret = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "MAXFLIPS = %d \t RET = %1.4f\n" maxflips ret
    return ret
end

function returns_versus_maxflips(element_size, maxflips, num_trajectories)
    env = make_env(element_size, maxflips)
    ret = GP.average_normalized_returns(env, num_trajectories)
    @printf "MAXFLIPS = %d \t RET = %1.4f\n" maxflips ret
    return ret
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

function PG.reset!(env::EdgeFlip.OrderedGameEnv; nflips = env.num_initial_flips)
    EdgeFlip.reset!(env, nflips = env.num_initial_flips)
end

# function PG.reset!(env::EdgeFlip.OrderedGameEnv)
#     EdgeFlip.reset!(env)
# end

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

function make_env(element_size, maxflips)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    num_edges = EdgeFlip.number_of_edges(mesh)
    d0 = fill(6, num_nodes)
    d0[mesh.bnd_nodes] .= 4

    env = EdgeFlip.OrderedGameEnv(mesh, 0, d0 = d0, maxflips = maxflips)

    return env
end

element_size = 0.3
maxflips = 25

env = make_env(element_size, maxflips)
# policy = Policy3L()

# PG.reset!(env)
# ep, econn, epairs = PG.state(env)
# l = PG.eval_single(policy, ep, econn, epairs)
# bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, 10, 1.0)
# l = PG.eval_batch(policy, bs[1], bs[2], bs[3])

# num_epochs = 1000
# batch_size = 50
# discount = 0.8

using BSON: @load
@load "results/models/new-edge-model/3L.bson" policy


learning_rate = 5e-4
optimizer = ADAM(learning_rate)
epochs, ret = PG.run_training_loop(env, policy, optimizer, batch_size, discount, num_epochs)

num_trajectories = 100
maxflip_range = 5:5:70
opt_ret = env.score - env.optimum_score
PG.reset!(env)
PG.single_trajectory_return(env, policy)
# PG.single_trajectory_normalized_return(env, policy)
nn_ret = [returns_versus_maxflips(policy, element_size, mf, num_trajectories) for mf in maxflip_range]


PG.reset!(env)
gd_ret = [returns_versus_maxflips(element_size, mf, num_trajectories) for mf in maxflip_range]


# gd_ret = GP.single_trajectory_return(env)
# gd_ret = GP.average_normalized_returns(env, num_trajectories)

# num_trajectories = 500
# nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
include("plot.jl")
plot_returns(maxflip_range, nn_ret, gd_ret = gd_ret, ylim = [0., 1])

# filename = "results/new-edge-model/3L-res-performance.png"
# plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75, 1], filename = filename)