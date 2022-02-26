using Flux
using Distributions: Categorical
using EdgeFlip
using Printf
include("edge_policy_gradient.jl")

PG = EdgePolicyGradient

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

function PG.reset!(env::EdgeFlip.GameEnv)
    EdgeFlip.reset!(env)
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
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)

discount = 0.7
learning_rate = 0.0001
batch_size = 5maxflips
num_epochs = 3000
num_trajectories = 100

policy = TwoLevelPolicy()

logits = policy(PG.state(env))

epoch_history, return_history = PG.run_training_loop(
    env,
    policy,
    batch_size,
    discount,
    num_epochs,
    learning_rate,
    print_every = 100,
);

# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy")

# include("plot.jl")
# filename = "results/extended-template/two-level-edge-template.png"
# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy", filename = filename)