using Flux
using EdgeFlip
using Printf
include("vertex_policy_gradient.jl")

PG = VertexPolicyGradient

mutable struct ConstantEnv
    state
    reward
    terminal
    function ConstantEnv()
        new([1.0 -1.0], 0.0, false)
    end
end

function PG.state(env::ConstantEnv)
    return env.state
end

function PG.step!(env::ConstantEnv, action)
    env.reward = action == 1 ? 0 : 1
    env.terminal = true
end

function PG.is_terminated(env::ConstantEnv)
    return env.terminal
end

function PG.reward(env::ConstantEnv)
    return env.reward
end

function PG.reset!(env::ConstantEnv)
    env.reward = 0
    env.terminal = false
end

struct Policy
    model
    function Policy()
        new(Dense(1,1))
    end
end

function (p::Policy)(s)
    return p.model(s)
end

Flux.@functor Policy

env = ConstantEnv()
policy = Policy()

learning_rate = 0.01
batch_size = 10
discount = 1.0
num_epochs = 1000
num_trajectories = 100

optimizer = ADAM(learning_rate)

old_probs = softmax(vec(policy(PG.state(env))))

PG.run_training_loop(env, policy, batch_size, discount, num_epochs, learning_rate)

new_probs = softmax(vec(policy(PG.state(env))))

println("Old probs : $old_probs")
println("New probs : $new_probs")