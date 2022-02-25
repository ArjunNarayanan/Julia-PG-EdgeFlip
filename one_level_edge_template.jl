using Flux
using Distributions: Categorical
using EdgeFlip
using Printf
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
    vertex_template_score, edge_template = state[1], state[2]

    ep = p.vmodel(vertex_template_score)

    es = edge_state(ep, edge_template, p.bmodel)
    es = p.emodel(es)

    logits = vec(es)

    return logits
end

Flux.@functor EdgePolicy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)

discount = 0.9
learning_rate = 0.001
batch_size = 32
num_epochs = 3000
num_trajectories = 100

policy = EdgePolicy()

old_bmodel = deepcopy(policy.bmodel)
epoch_history, return_history = PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    discount,
    num_epochs,
    learning_rate,
    num_trajectories,
    estimate_every = 100,
);
new_bmodel = deepcopy(policy.bmodel)



# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy")

# epoch_history .+= 3000
# include("plot.jl")
# filename = "results/extended-template/one-level-edge-template.png"
# plot_history(epoch_history, return_history, optimum = 0.91, opt_label = "greedy", filename = filename)