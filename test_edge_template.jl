using Flux
using Distributions: Categorical
using EdgeFlip
import EdgeFlip: state, step!, is_terminated, reward, reset!
using Printf
include("global_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

function edge_state(edge_potentials, boundary_vector, active_edgeid, env)
    edgeid = env.active_edge_to_edge[active_edgeid]
    state = vcat([env.active_edges[e] ? edge_potentials[:,e] : boundary_vector for e in env.edge_template[:,edgeid]]...)
    return state
end

struct ExtPolicy
    vmodel
    emodel
    bmodel
end

function (p::ExtPolicy)(env::EdgeFlip.GameEnv)
    num_actions = EdgeFlip.number_of_actions(env)
    epotentials = p.vmodel(EdgeFlip.state(env))
    estate = hcat([edge_state(epotentials,p.bmodel,i,env) for i in 1:num_actions]...)
    return vec(p.emodel(estate))
end

Flux.@functor ExtPolicy

vmodel = Dense(4,4)
emodel = Dense(20,1)
bmodel = Flux.glorot_uniform(4)
policy = ExtPolicy(vmodel,emodel,bmodel)


nref = 0
nflips = 2
env = EdgeFlip.GameEnv(nref,nflips)

learning_rate = 0.1
batch_size = 32
num_epochs = 1000
maxsteps = ceil(Int, 1.2nflips)
num_trajectories = 100


epoch_history, return_history = PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate,
    maxsteps,
    num_trajectories,
    estimate_every = 100,
)