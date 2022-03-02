using Flux
using EdgeFlip
include("supervised_greedy_training.jl")
include("vertex_policy_gradient.jl")

SV = Supervised
PG = VertexPolicyGradient

SV.state(env::EdgeFlip.GameEnv) = EdgeFlip.vertex_template_score(env)

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
        model = Chain(Dense(4, 10, relu), Dense(10, 10, relu), Dense(10, 10, relu), Dense(10, 1))
        # model = Chain(Dense(4,1))
        new(model)
    end
end

function (p::VertexPolicy)(s)
    return p.model(s)
end

Flux.@functor VertexPolicy


nref = 1
nflips = 8
maxflips = ceil(Int,1.2nflips)
batch_size = 5maxflips
num_supervised_epochs = 1000
num_rl_epochs = 1000
rl_learning_rate = 0.001
discount = 0.9
env = EdgeFlip.GameEnv(nref,nflips,maxflips=maxflips)

policy = VertexPolicy()
optimizer = ADAM(0.01)

sv_loss = SV.run_training_loop(env, policy, optimizer, batch_size, num_supervised_epochs)
# rl_epochs, rl_loss = PG.run_training_loop(env, policy, batch_size, discount, num_rl_epochs, rl_learning_rate)


# include("plot.jl")
# plot_history(1:num_supervised_epochs,sv_loss,ylim=[0,5])

# include("greedy_policy.jl")

# num_trajectories = 500
# ret = PG.average_normalized_returns(env, policy, num_trajectories)
# gret = GreedyPolicy.average_normalized_returns(env, num_trajectories)