using Printf
using Flux
include("vertex_policy_gradient.jl")
include("greedy_policy.jl")

PG = VertexPolicyGradient
GP = GreedyPolicy

mutable struct VertexTemplate
    state::Any
    reward::Any
    terminal::Any
    function VertexTemplate(state)
        reward = 0
        terminal = false
        new(state, reward, terminal)
    end
end

function VertexTemplate()
    state = rand(-1:1, 4)
    return VertexTemplate(state)
end

function PG.state(env::VertexTemplate)
    return reshape(env.state, :, 1)
end

function PG.step!(env::VertexTemplate, action)
    vals = env.state .+ [1, -1, 1, -1]
    old_score = sum(abs.(env.state))
    new_score = sum(abs.(vals))

    env.reward = action == 2 ? old_score - new_score : 0
    env.state .= vals
    env.terminal = true
end

function PG.is_terminated(env::VertexTemplate)
    return env.terminal
end

function PG.reward(env::VertexTemplate)
    return env.reward
end

function PG.reset!(env::VertexTemplate)
    env.state .= rand(-1:1, 4)
    env.reward = 0.0
    env.terminal = false
end

function GP.greedy_action(env::VertexTemplate)
    vals = env.state .+ [1, -1, 1, -1]
    old_score = sum(abs.(env.state))
    new_score = sum(abs.(vals))
    action = old_score > new_score ? 2 : 1
    return action
end

function GP.step!(env::VertexTemplate, action)
    PG.step!(env, action)
end

function GP.is_terminated(env::VertexTemplate)
    return PG.is_terminated(env)
end

function GP.reward(env::VertexTemplate)
    return PG.reward(env)
end

function GP.reset!(env::VertexTemplate)
    PG.reset!(env)
end

struct Policy
    model::Any
    function Policy()
        # new(Chain(Dense(4,20,relu), Dense(20,20,relu), Dense(20,1)))
        new(Dense(4, 2))
    end
end

function (p::Policy)(s::AbstractArray{T,2}) where {T}
    return transpose(p.model(s))
end

function (p::Policy)(s::AbstractArray{T,3}) where {T}
    return reshape(p.model(s), 1, 2, :)
end

function find_non_greedy(env, policy)
    found = false
    while !found
        PG.reset!(env)
        a = argmax(vec(policy(PG.state(env))))
        ga = GP.greedy_action(env)

        PG.step!(env, a)
        r = PG.reward(env)

        PG.step!(env, ga)
        gr = PG.reward(env)

        found = r != gr
    end
end

Flux.@functor Policy

learning_rate = 0.01
batch_size = 100
discount = 1.0
num_epochs = 1000
num_trajectories = 10000

policy = Policy()
env = VertexTemplate()

bs, ba, bw, ret = PG.collect_batch_trajectories(env, policy, batch_size, discount)

# policy.model.weight .= [-8.2 -7.6 8.5 8.8
#                          8.8 8.0 -8.5 -8.2]
# policy.model.bias .= [12,-12]

PG.run_training_loop(env, policy, batch_size, discount, num_epochs, learning_rate)
nn_ret = PG.average_returns(env, policy, num_trajectories)
# gd_ret = GP.average_returns(env, num_trajectories)

# find_non_greedy(env, policy)

# state = PG.state(env)
# nnprobs = softmax(vec(policy(PG.state(env))))
# ga = GP.greedy_action(env)
# w,b = params(policy)