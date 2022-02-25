using BenchmarkTools
using Flux
using Distributions: Categorical
using EdgeFlip
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

struct EdgePolicy
    vmodel::Any
    emodel::Any
    bmodel::Any
    function EdgePolicy()
        vmodel = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))

        emodel = Chain(Dense(20, 10, relu), Dense(10, 10, relu), Dense(10, 1, relu))
        bmodel = Flux.glorot_uniform(4)

        new(vmodel, emodel, bmodel)
    end
end

function (p::EdgePolicy)(state)
    vertex_template_score, edge_template = state
    num_edges = size(vertex_template_score, 2)

    ep = p.vmodel(vertex_template_score)
    ep = cat(ep, p.bmodel, dims = 2)

    es = edge_state(ep, edge_template, num_edges)
    logits = p.emodel(es)

    return logits
end

function (p::EdgePolicy)(vertex_template_score, edge_template, num_batches)
    num_edges = size(vertex_template_score, 2)

    ep = p.vmodel(vertex_template_score)
    ep = cat(ep, repeat(p.bmodel, inner = (1, 1, num_batches)), dims = 2)

    es = batch_edge_state(ep, edge_template, num_edges)
    es = p.emodel(es)

    logits = es

    return logits
end

Flux.@functor EdgePolicy


function single_edge_state(edgeid, edge_potentials, edge_template)
    es = vec(edge_potentials[:, edge_template[:, edgeid]])
    return es
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

function get_gradient(vs, et, policy, actions, weights)
    theta = Flux.params(policy)
    grads = Flux.gradient(theta) do
        loss = PG.policy_gradient_loss(vs, et, policy, actions, weights)
    end
    return grads
end

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)


learning_rate = 0.1
batch_size = 32
discount = 0.5
num_epochs = 1000
num_trajectories = 100


policy = EdgePolicy()
optimizer = ADAM(learning_rate)


vs, et, actions, weights, ret =
    PG.collect_batch_trajectories(env, policy, batch_size, discount)
@btime PG.collect_batch_trajectories($env, $policy, $batch_size, $discount)

@btime get_gradient($vs, $et, $policy, $actions, $weights)


# PG.run_training_loop(env, policy, batch_size, discount, num_epochs, learning_rate)
# @btime PG.run_training_loop(
#     $env,
#     $policy,
#     $batch_size,
#     $discount,
#     $num_epochs,
#     $learning_rate,
# )
