using BenchmarkTools
using Flux
using Distributions: Categorical
using EdgeFlip
include("old_global_policy_gradient.jl")

PG = PolicyGradient

function PG.state(env::EdgeFlip.GameEnv)
    vs = EdgeFlip.vertex_template_score(env)
    et = EdgeFlip.edge_template(env)

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

    ep = p.vmodel(vertex_template_score)

    es = edge_state(ep, edge_template, p.bmodel)
    logits = vec(p.emodel(es))

    return logits
end

Flux.@functor EdgePolicy


function single_edge_state(edgeid, edge_potentials, edge_template, boundary_values)
    nbr_edges = edge_template[:, edgeid]
    es = vcat([e == 0 ? boundary_values : edge_potentials[:, e] for e in nbr_edges]...)
    return es
end

function edge_state(edge_potentials, edge_template, boundary_values)
    es = hcat(
        [
            single_edge_state(e, edge_potentials, edge_template, boundary_values) for
            e = 1:size(edge_potentials, 2)
        ]...,
    )
    return es
end

function get_gradient(states, policy, actions, weights)
    theta = Flux.params(policy)
    grads = Flux.gradient(theta) do
        loss = PG.policy_gradient_loss(states, policy, actions, weights)
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

s = PG.state(env)
logits = policy(s)

# states, actions, weights, ret =
#     PG.collect_batch_trajectories(env, policy, batch_size, discount)
# PG.policy_gradient_loss(states, policy, actions, weights)
# @btime get_gradient($states, $policy, $actions, $weights)

# @btime PG.run_training_loop(
#     $env,
#     $policy,
#     $batch_size,
#     $discount,
#     $num_epochs,
#     $learning_rate,
# )
