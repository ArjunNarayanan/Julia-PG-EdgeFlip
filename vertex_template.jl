using Flux
using EdgeFlip
using Printf
include("vertex_policy_gradient.jl")

PG = VertexPolicyGradient

function PG.state(env::EdgeFlip.GameEnv)
    return EdgeFlip.vertex_template_score(env)
end

function PG.step!(env::EdgeFlip.GameEnv, action)
    EdgeFlip.step!(env, action)
end

function PG.is_terminated(env::EdgeFlip.GameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.GameEnv)
    return EdgeFlip.reward(env)
end

function PG.reset!(env::EdgeFlip.GameEnv)
    EdgeFlip.reset!(env)
end

function PG.score(env::EdgeFlip.GameEnv)
    return EdgeFlip.score(env)
end

struct VertexPolicy
    model::Any
    function VertexPolicy()
        # model = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 1))
        model = Chain(Dense(4,1))
        new(model)
    end
end

function (p::VertexPolicy)(s)
    return p.model(s)
end

Flux.@functor VertexPolicy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)

learning_rate = 0.01
batch_size = 10maxflips
discount = 1.0
num_epochs = 1000
num_trajectories = 100

policy = VertexPolicy()


epoch_history, return_history = PG.run_training_loop(
    env,
    policy,
    batch_size,
    discount,
    num_epochs,
    learning_rate,
    print_every = 100,
)

# include("plot.jl")

num_test_trajectories = 1000
nn_avg = PG.average_returns(env, policy, num_test_trajectories)

include("greedy_policy.jl")
greedy_avg = GreedyPolicy.average_returns(env, num_test_trajectories)
# @printf "NN MEAN : %2.3f\n" nn_avg
# @printf "GD MEAN : %2.3f\n" greedy_avg