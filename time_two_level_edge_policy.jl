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

struct TwoLevelEdgePolicy
    vmodel::Any
    emodel1::Any
    bmodel1::Any
    emodel2::Any
    bmodel2::Any
    function TwoLevelEdgePolicy()
        vmodel = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))

        emodel1 = Chain(Dense(20, 10, relu), Dense(10, 10, relu), Dense(10, 4, relu))
        bmodel1 = Flux.glorot_uniform(4)

        emodel2 = Chain(Dense(20, 10, relu), Dense(10, 10, relu), Dense(10, 1))
        bmodel2 = Flux.glorot_uniform(4)

        new(vmodel, emodel1, bmodel1, emodel2, bmodel2)
    end
end

function (p::TwoLevelEdgePolicy)(state)
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

function (p::TwoLevelEdgePolicy)(vertex_template_score, edge_template, num_batches)
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

Flux.@functor TwoLevelEdgePolicy


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


policy = TwoLevelEdgePolicy()
optimizer = ADAM(learning_rate)

# s = PG.state(env)
# logits = policy(s)


# vs, et, actions, weights, ret =
#     PG.collect_batch_trajectories(env, policy, batch_size, discount)
# @btime PG.collect_batch_trajectories($env, $policy, $batch_size, $discount)

# get_gradient(vs, et, policy, actions, weights)
# @btime get_gradient($vs, $et, $policy, $actions, $weights)

# PG.run_training_loop(env, policy, batch_size, discount, num_epochs, learning_rate)
@btime PG.run_training_loop(
    $env,
    $policy,
    $batch_size,
    $discount,
    $num_epochs,
    $learning_rate,
)
