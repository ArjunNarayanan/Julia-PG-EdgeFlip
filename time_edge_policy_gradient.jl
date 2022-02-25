using BenchmarkTools
using Flux
using Distributions: Categorical
using EdgeFlip
include("global_policy_gradient.jl")

function PolicyGradient.state(env::EdgeFlip.GameEnv)
    vs = EdgeFlip.vertex_template_score(env)
    et = EdgeFlip.edge_template(env)
    return vs, et
end

function PolicyGradient.step!(env::EdgeFlip.GameEnv, action)
    EdgeFlip.step!(env, action)
end

function PolicyGradient.is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function PolicyGradient.reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function PolicyGradient.reset!(env::EdgeFlip.GameEnv)
    EdgeFlip.reset!(env)
end

function PolicyGradient.score(env::EdgeFlip.GameEnv)
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

function (p::EdgePolicy)(s)
    vertex_template_score, edge_template = s[1], s[2]
    ep = p.vmodel(vertex_template_score)

    es = old_edge_state(ep, edge_template, p.bmodel)
    es = p.emodel(es)

    logits = vec(es)

    return logits
end

function (p::EdgePolicy)(vertex_template_score, edge_template)
    @assert size(vertex_template_score, 3) == size(edge_template, 3)
    num_batches = size(vertex_template_score, 3)
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

function old_single_edge_state(edgeid, edge_potentials, edge_template, boundary_values)
    nbr_edges = edge_template[:, edgeid]
    es = vcat([e == 0 ? boundary_values : edge_potentials[:, e] for e in nbr_edges]...)
    return es
end

function old_edge_state(edge_potentials, edge_template, boundary_values)
    es = hcat(
        [
            old_single_edge_state(e, edge_potentials, edge_template, boundary_values)
            for e = 1:size(edge_potentials, 2)
        ]...,
    )
    return es
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

# function (p::EdgePolicy)(vertex_template_score, edge_template)

#     ep = p.vmodel(vertex_template_score)

#     es = old_edge_state(ep, edge_template, p.bmodel)
#     es = p.emodel(es)

#     logits = vec(es)

#     return logits
# end

function get_gradient(vertex_template_score, edge_template, policy, actions, weights)
    θ = Flux.params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = PolicyGradient.edge_policy_gradient_loss(vertex_template_score, edge_template, policy, actions, weights)
    end
    return grads
end

function old_get_gradient(states, policy, actions, weights)
    θ = Flux.params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = PolicyGradient.old_policy_gradient_loss(states, policy, actions, weights)
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



vs, et = PolicyGradient.state(env)

mvs = repeat(vs, inner = (1, 1, batch_size))
met = repeat(et, inner = (1, 1, batch_size))
met[met .== 0] .= 43
actions = rand(1:42,32)
weights = rand(32)

grads = get_gradient(mvs, met, policy, actions, weights)
@btime get_gradient($mvs, $met, $policy, $actions, $weights)

# states, actions, weights, ret =
#     PolicyGradient.old_collect_batch_trajectories(env, policy, batch_size, discount)

# @btime PolicyGradient.old_collect_batch_trajectories($env, $policy, $batch_size, $discount)

# grads = old_get_gradient(states, policy, actions, weights)
# @btime old_get_gradient($states, $policy, $actions, $weights)

# theta = Flux.params(policy)
# Flux.update!(optimizer, theta, grads)

# @btime Flux.update!($optimizer, $theta, $grads)