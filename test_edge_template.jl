using Flux
using Distributions: Categorical
using EdgeFlip
# import EdgeFlip: state, step!, is_terminated, reward, reset!
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

function edge_state(edgeid, edge_potentials, edge_template, boundary_vector)
    state = vcat([e != 0 ? edge_potentials[:,e] : boundary_vector for e in edge_template[:,edgeid]]...)
    return state
end

struct ExtPolicy
    vmodel
    emodel
    bmodel
end

function (p::ExtPolicy)(state)
    vertex_template_score, edge_template = state[1], state[2]

    num_actions = size(vertex_template_score,2)
    epotentials = p.vmodel(vertex_template_score)
    epotentials = hcat(epotentials,p.bmodel)

    logits = [p.emodel(reshape(view(epotentials,:,edge_template[:,e]),:))[1] for e in 1:num_actions]

    logits = -Inf*ones(num_actions)
    logits[1] = 1.0

    return logits
end

Flux.@functor ExtPolicy

vmodel = Dense(4,1)
emodel = Dense(5,1)
bmodel = Flux.glorot_uniform(1)
policy = ExtPolicy(vmodel,emodel,bmodel)


nref = 0
nflips = 1
env = EdgeFlip.GameEnv(nref,nflips,fixed_reset=true)


learning_rate = 0.1
batch_size = 1
num_epochs = 10000
maxsteps = 1 # ceil(Int, 1.2nflips)
num_trajectories = 100

batch_states, batch_actions, batch_weights, avg_return = PolicyGradient.collect_batch_trajectories(env,policy,batch_size)
println(batch_actions)
println(batch_weights)
PolicyGradient.reset!(env)

# optimizer = ADAM(learning_rate)
# PolicyGradient.step_epoch(env,policy,optimizer,batch_size)

# epoch_history, return_history = PolicyGradient.run_training_loop(
#     env,
#     policy,
#     batch_size,
#     num_epochs,
#     learning_rate,
#     maxsteps,
#     num_trajectories,
#     estimate_every = 100,
# )