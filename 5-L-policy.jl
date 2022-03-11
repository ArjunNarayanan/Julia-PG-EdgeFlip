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

    vs = Flux.Float32.(vs)

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

function SV.reset!(env)
    PG.reset!(env)
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

struct EdgeModel
    emodel::Any
    bmodel::Any
    function EdgeModel(in_channels, hidden_channels, out_channels)
        emodel = Chain(
            Dense(5in_channels, hidden_channels, relu),
            Dense(hidden_channels, out_channels, relu),
        )
        bmodel = Flux.glorot_uniform(in_channels)
        new(emodel, bmodel)
    end
end

function (p::EdgeModel)(edge_potentials, edge_template)
    num_edges = size(edge_template, 2)

    ep = cat(edge_potentials, p.bmodel, dims = 2)
    es = edge_state(ep, edge_template, num_edges)

    return p.emodel(es)
end

function (p::EdgeModel)(edge_potentials, edge_template, num_batches)
    num_edges = size(edge_template, 2)

    ep = cat(edge_potentials, repeat(p.bmodel, inner = (1,1,num_batches)), dims = 2)
    es = batch_edge_state(ep, edge_template, num_edges)

    return p.emodel(es)
end

Flux.@functor EdgeModel

struct FiveLevelPolicy
    vmodel::Any
    emodel1::Any
    emodel2::Any
    emodel3::Any
    emodel4::Any
    emodel5::Any
    # linear
    function FiveLevelPolicy()
        vmodel = Chain(Dense(4, 4, relu))

        emodel1 = EdgeModel(4, 4, 4)
        emodel2 = EdgeModel(4, 4, 4)
        emodel3 = EdgeModel(4, 4, 4)
        emodel4 = EdgeModel(4, 4, 4)
        emodel5 = EdgeModel(4, 4, 1)
        # linear = Dense(4,1)

        new(vmodel, emodel1, emodel2, emodel3, emodel4, emodel5)
    end
end

function (p::FiveLevelPolicy)(state)
    vertex_template_score, edge_template = state

    ep = p.vmodel(vertex_template_score)

    ep = p.emodel1(ep, edge_template)
    ep = p.emodel2(ep, edge_template)
    ep = p.emodel3(ep, edge_template)
    ep = p.emodel4(ep, edge_template)
    ep = p.emodel5(ep, edge_template)
    logits = ep

    return logits
end

function (p::FiveLevelPolicy)(states, num_batches)
    vertex_template_score, edge_template = states

    ep = p.vmodel(vertex_template_score)

    ep = p.emodel1(ep, edge_template, num_batches)
    ep = p.emodel2(ep, edge_template, num_batches)
    ep = p.emodel3(ep, edge_template, num_batches)
    ep = p.emodel4(ep, edge_template, num_batches)
    ep = p.emodel5(ep, edge_template, num_batches)
    logits = ep

    return logits
end

Flux.@functor FiveLevelPolicy

nref = 1
env = EdgeFlip.GameEnv(nref, 0)
num_actions = EdgeFlip.number_of_actions(env)

batch_size = 100
num_supervised_epochs = 500
sv_learning_rate = 0.001

# policy = FiveLevelPolicy()

# sv_loss = SV.run_edge_training_loop(
#     env,
#     policy,
#     batch_size,
#     num_supervised_epochs,
#     sv_learning_rate,
# )

# num_trajectories = 500
# nflip_range = 1:5:42
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# normalized_nflips = nflip_range ./ num_actions

num_rl_epochs = 1000
rl_learning_rate = 1e-4
discount = 0.9

rl_epochs, rl_loss =
    PG.run_training_loop(env, policy, batch_size, discount, num_rl_epochs, rl_learning_rate)
nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1])
# plot_returns(
#     normalized_nflips,
#     nn_ret,
#     gd_ret = gd_ret,
#     ylim = [0.75, 1],
#     filename = "results/supervised/edge-policy/ep-2-rl-vs-gd-random-reset-10000.png",
# )

