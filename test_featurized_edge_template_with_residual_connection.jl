using Flux
using Distributions: Categorical
using EdgeFlip
using Printf
include("global_policy_gradient.jl")
include("plot.jl")

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

struct EdgePolicy
    vmodel::Any
    emodel1::Any
    emodel2::Any
    rmodel::Any
    bmodel::Any
    function EdgePolicy()
        vmodel = Dense(4, 4, relu)
        emodel1 = Dense(20, 4, relu)
        emodel2 = Dense(20, 4, relu)
        rmodel = Dense(8, 1)
        bmodel = Flux.glorot_uniform(4)
        new(vmodel, emodel1, emodel2, rmodel, bmodel)
    end
end

function (p::EdgePolicy)(state)
    vertex_template_score, edge_template = state[1], state[2]

    ep = p.vmodel(vertex_template_score)

    es = edge_state(ep, edge_template, p.bmodel)
    es = p.emodel1(es)

    es = edge_state(es, edge_template, p.bmodel)
    es = p.emodel2(es)

    vs = eltype(ep).(vertex_template_score)
    fs = vcat(vs, es)

    logits = vec(p.rmodel(fs))

    return logits
end

Flux.@functor EdgePolicy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)

learning_rate = 0.1
batch_size = 32
num_epochs = 1000
num_trajectories = 100

policy = EdgePolicy()
# logits = policy(PolicyGradient.state(env))

epoch_history, return_history = PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate,
    num_trajectories,
    estimate_every = 100,
);

# avg_ret = PolicyGradient.average_normalized_returns(env, policy, 1000)

# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy")

# include("plot.jl")
# filename = "results/extended-template/residual-featurized-edge-template.png"
# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy", filename = filename)