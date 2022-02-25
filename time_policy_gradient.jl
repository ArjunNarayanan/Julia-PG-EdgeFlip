using BenchmarkTools
using Flux
using Distributions: Categorical
using EdgeFlip
include("global_policy_gradient.jl")

function PolicyGradient.state(env::EdgeFlip.GameEnv)
    return EdgeFlip.vertex_template_score(env)
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

struct Policy
    model::Any
    function Policy(model)
        new(model)
    end
end

function (p::Policy)(s)
    return p.model(s)
end

function get_gradient(states, policy, actions, weights)
    θ = Flux.params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = PolicyGradient.policy_gradient_loss(states, policy, actions, weights)
    end
    return grads
end

function get_ndim_gradient(states, policy, actions, weights)
    θ = Flux.params(policy)
    loss = 0.0
    grads = Flux.gradient(θ) do
        loss = PolicyGradient.ndim_policy_gradient_loss(states, policy, actions, weights)
    end
    return grads
end

Flux.@functor Policy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)


learning_rate = 0.1
batch_size = 32
discount = 0.5
num_epochs = 1000
num_trajectories = 100


policy = Policy(Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 1)))
optimizer = ADAM(learning_rate)


# states = [env.vertex_template_score for i = 1:32]
# mstates = cat(states...,dims=3)
# actions = rand(1:42,32)
# weights = rand(32)

# grads = get_gradient(states, policy, actions, weights)
# ngrads = get_ndim_gradient(mstates, policy, actions, weights)

# @btime get_gradient($states, $policy, $actions, $weights)
# @btime get_ndim_gradient($mstates, $policy, $actions, $weights)


# logits = policy(PolicyGradient.state(env))
# probs = softmax(logits,dims=2)
# c = Categorical(reshape(probs,:))



PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    discount,
    num_epochs,
    learning_rate,
    print_every = 100,
)
# @btime PolicyGradient.run_training_loop(
#     $env,
#     $policy,
#     $batch_size,
#     $discount,
#     $num_epochs,
#     $learning_rate,
#     print_every = 100,
# )
