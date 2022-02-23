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

function evaluate_edge(edgeid, ep, et, bvec, emodel)
    nbr_edges = et[:,edgeid]
    es = vcat([e == 0 ? bvec : ep[:,e] for e in nbr_edges]...)
    return emodel(es)[1]
end

struct EdgePolicy
    vmodel::Any
    emodel::Any
    bmodel::Any
    function EdgePolicy()
        vmodel = Chain(Dense(4, 4, relu),Dense(4,4))
        emodel = Dense(20, 1)
        bmodel = Flux.glorot_uniform(4)
        new(vmodel, emodel, bmodel)
    end
end

function (p::EdgePolicy)(state)
    vertex_template_score, edge_template = state[1], state[2]

    ep = p.vmodel(vertex_template_score)
    logits = [evaluate_edge(e, ep, edge_template, p.bmodel, p.emodel) for e in 1:size(edge_template,2)]

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
old_bmodel = deepcopy(policy.bmodel)
epoch_history, return_history = PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate,
    num_trajectories,
    estimate_every = 100,
);
new_bmodel = deepcopy(policy.bmodel)

# include("plot.jl")
# filename = "results/extended-template/nl-featurized-edge-template.png"
# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy", filename = filename)