using Printf
using Flux
using EdgeFlip
include("supervised_greedy_training.jl")
include("edge_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

SV = Supervised
PG = EdgePolicyGradient

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t RET = %1.3f\n" nflips avg
    return avg
end

function returns_versus_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t RET = %1.3f\n" nflips avg
    return avg
end

SV.state(env::EdgeFlip.GameEnv) = PG.state(env)

function PG.state(env::EdgeFlip.GameEnv)
    vs = EdgeFlip.vertex_template_score(env)
    et = EdgeFlip.edge_template(env)
    num_edges = size(vs, 2)
    et[et.==0] .= num_edges + 1
    return vs, et
end

function PG.step!(env::EdgeFlip.GameEnv, action)
    EdgeFlip.step!(env, action)
end

function PG.is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function PG.reset!(env::EdgeFlip.GameEnv; nflips = rand(1:42))
    maxflips = ceil(Int, 1.2nflips)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.score(env::EdgeFlip.GameEnv)
    return EdgeFlip.score(env)
end

function single_edge_state(edgeid, edge_potentials, edge_template)
    es = vec(edge_potentials[:, edge_template[:, edgeid]])
    return es
end

function single_edge_state(edge_potentials, edgeids)
    return vec(edge_potentials[:, edgeids])
end

function edge_state(edge_potentials, edge_template, num_edges)
    es = [single_edge_state(e, edge_potentials, edge_template) for e = 1:num_edges]
    return hcat(es...)
end

function batch_edge_state(edge_potentials, edge_template, num_edges)
    @assert size(edge_potentials, 3) == size(edge_template, 3)
    num_batches = size(edge_template, 3)
    es = [
        edge_state(edge_potentials[:, :, b], edge_template[:, :, b], num_edges) for
        b = 1:num_batches
    ]
    return cat(es..., dims = 3)
end

struct TwoLevelPolicy
    vmodel::Any
    emodel1::Any
    bmodel1::Any
    emodel2::Any
    bmodel2::Any
    function TwoLevelPolicy()
        vmodel = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))

        emodel1 = Chain(Dense(20, 10, relu), Dense(10, 10, relu), Dense(10, 4, relu))
        bmodel1 = Flux.glorot_uniform(4)

        emodel2 = Chain(Dense(20, 10, relu), Dense(10, 10, relu), Dense(10, 1, relu))
        bmodel2 = Flux.glorot_uniform(4)

        new(vmodel, emodel1, bmodel1, emodel2, bmodel2)
    end
end

function (p::TwoLevelPolicy)(state)
    vertex_template_score, edge_template = state
    num_edges = size(vertex_template_score, 2)

    ep = p.vmodel(vertex_template_score)
    ep = cat(ep, p.bmodel1, dims = 2)

    es = edge_state(ep, edge_template, num_edges)
    es = p.emodel1(es)

    es = cat(es, p.bmodel2, dims = 2)
    es = edge_state(es, edge_template, num_edges)
    
    logits = p.emodel2(es)

    return logits
end

function (p::TwoLevelPolicy)(states, num_batches)
    vertex_template_score, edge_template = states
    num_edges = size(vertex_template_score, 2)

    ep = p.vmodel(vertex_template_score)
    ep = cat(ep, repeat(p.bmodel1, inner = (1, 1, num_batches)), dims = 2)

    es = batch_edge_state(ep, edge_template, num_edges)
    es = p.emodel1(es)

    es = cat(es, repeat(p.bmodel2, inner = (1, 1, num_batches)), dims = 2)
    es = batch_edge_state(es, edge_template, num_edges)
    
    logits = p.emodel2(es)

    return logits
end

Flux.@functor TwoLevelPolicy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
batch_size = 5maxflips
num_supervised_epochs = 500
sv_learning_rate = 0.001

env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
num_actions = EdgeFlip.number_of_actions(env)

policy = TwoLevelPolicy()

sv_loss = SV.run_edge_training_loop(
    env,
    policy,
    batch_size,
    num_supervised_epochs,
    sv_learning_rate,
)


# plot_history(
#     1:num_supervised_epochs,
#     sv_loss,
#     ylim = [1, 5],
#     ylabel = "cross entropy loss",
#     filename = "results/supervised/edge-policy/edge-2-sv-loss.png",
# )

# num_trajectories = 500
# nflip_range = 1:5:42
# normalized_nflips = nflip_range ./ num_actions
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]


num_rl_epochs = 5000
rl_learning_rate = 5e-4
discount = 1.0
rl_epochs, rl_loss =
    PG.run_training_loop(env, policy, batch_size, discount, num_rl_epochs, rl_learning_rate)

nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1])
# plot_returns(
#     normalized_nflips,
#     nn_ret,
#     gd_ret = gd_ret,
#     ylim = [0.75, 1],
#     filename = "results/supervised/edge-policy/ep-2-rl-vs-gd.png",
# )
