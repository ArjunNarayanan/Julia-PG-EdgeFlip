using Flux
using Distributions: Categorical
using EdgeFlip
using Printf
include("global_policy_gradient.jl")
include("greedy_policy.jl")
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

function edge_state(edge_potentials, edge_template, bval)
    es = [e == 0 ? bval : edge_potentials[e] for e in edge_template]
    return es
end

struct ExtPolicy
    vmodel
    emodel
    bmodel
end

function (p::ExtPolicy)(state)
    vertex_template_score, edge_template = state[1], state[2]

    ep = p.vmodel(vertex_template_score)
    es = edge_state(ep, edge_template, p.bmodel[1])

    T = eltype(es)
    vs = T.(vertex_template_score)
    ext_state = vcat(vs,es)
    logits = vec(p.emodel(ext_state))
    # logits = vec(ep)

    return logits
end

Flux.@functor ExtPolicy

vmodel = Dense(4,1)
# emodel = Dense(9,1)
emodel = Chain(Dense(9,4,relu),Dense(4,1))
bmodel = Flux.glorot_uniform(1)
policy = ExtPolicy(vmodel,emodel,bmodel)

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref,nflips,fixed_reset=false, maxflips = maxflips)

# learning_rate = 0.1
# batch_size = 32
# num_epochs = 1000
num_trajectories = 1000

gd_avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
println("Greedy avg = $gd_avg")



# epoch_history, return_history = PolicyGradient.run_training_loop(
#     env,
#     policy,
#     batch_size,
#     num_epochs,
#     learning_rate,
#     num_trajectories,
#     estimate_every = 100,
# );