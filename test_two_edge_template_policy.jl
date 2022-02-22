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
    if action > EdgeFlip.number_of_actions(env)
        EdgeFlip.step!(env)
    else
        EdgeFlip.step!(env, action)
    end
end

function PolicyGradient.is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function PolicyGradient.reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function PolicyGradient.reset!(env::EdgeFlip.GameEnv; vs = rand(-3:3,5))
    t = [
        1 2 5
        2 4 5
        2 3 4
    ]
    p = rand(5, 2)
    mesh = EdgeFlip.Mesh(p, t)

    env.mesh = mesh
    env.d0 .= mesh.d - vs
    env.vertex_score .= vs
    env.vertex_template = EdgeFlip.make_vertex_template(mesh.edges, mesh.t, mesh.t2t, mesh.t2e)
    env.vertex_template_score = EdgeFlip.make_vertex_template_score(env.vertex_score, env.vertex_template)
    num_edges = EdgeFlip.number_of_edges(mesh)
    env.edge_template = EdgeFlip.make_edge_template(mesh.t2t, mesh.t2e, num_edges)
    env.score = EdgeFlip.global_score(env.vertex_score)
    env.reward = 0
    env.is_terminated = env.score == 0 ? true : false
end

function PolicyGradient.score(env::EdgeFlip.GameEnv)
    return EdgeFlip.score(env)
end

function edge_state(edge_potentials, edge_template, boundary_value)
    state = [e == 0 ? boundary_value : edge_potentials[e] for e in edge_template]
    return state
end

struct ExtPolicy
    vmodel::Any
    emodel::Any
    bmodel::Any
    no_flip::Any
    function ExtPolicy()
        vmodel = Dense(4, 1)
        emodel = Dense(9, 1)
        bmodel = Flux.rand()
        no_flip = Flux.rand()
        new(vmodel, emodel, bmodel, no_flip)
    end
end

function (p::ExtPolicy)(state)
    vertex_template_score, edge_template = state[1], state[2]

    ep = policy.vmodel(vertex_template_score)
    
    # es = edge_state(ep, edge_template, p.bmodel)
    # s = vcat(float(vertex_template_score), es)
    # logits = vcat(vec(p.emodel(s)), p.no_flip)
    logits = vcat(vec(ep), p.no_flip)

    return logits
end

Flux.@functor ExtPolicy

function two_edge_game_env(; vertex_score = rand(-3:3, 5))
    t = [
        1 2 5
        2 4 5
        2 3 4
    ]
    p = rand(5, 2)
    mesh = EdgeFlip.Mesh(p, t)

    env = EdgeFlip.GameEnv(mesh, 0, d0 = mesh.d - vertex_score)
    return env
end

policy = ExtPolicy()
env = two_edge_game_env(vertex_score = zeros(Int,5))


learning_rate = 0.1
batch_size = 32
num_epochs = 1000
maxsteps = 2
num_trajectories = 100


# PolicyGradient.reset!(env)
# logits = policy(PolicyGradient.state(env))
# PolicyGradient.run_training_loop(env, policy, batch_size, num_epochs, learning_rate, maxsteps, num_trajectories)