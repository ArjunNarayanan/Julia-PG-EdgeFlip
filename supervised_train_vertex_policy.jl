using Printf
using Flux
using EdgeFlip
include("supervised_greedy_training.jl")
include("vertex_policy_gradient.jl")
include("greedy_policy.jl")

SV = Supervised
PG = VertexPolicyGradient

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t RET = %1.3f\n" nflips avg
    return avg
end

function returns_versus_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t RET = %1.3f\n" nflips avg
    return avg
end

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
        model = Chain(
            Dense(4, 4, relu),
            Dense(4, 4, relu),
            Dense(4, 4, relu),
            Dense(4, 1),
        )
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
maxflips = ceil(Int, 1.2nflips)
batch_size = 5maxflips
num_supervised_epochs = 500
num_rl_epochs = 5000
sv_learning_rate = 0.001
rl_learning_rate = 0.001
discount = 0.9

env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
num_actions = EdgeFlip.number_of_actions(env)

policy = VertexPolicy()

sv_loss = SV.run_training_loop(env, policy, batch_size, num_supervised_epochs, sv_learning_rate)
rl_epochs, rl_loss =
    PG.run_training_loop(env, policy, batch_size, discount, num_rl_epochs, rl_learning_rate)


# include("plot.jl")
# plot_history(
#     1:num_supervised_epochs,
#     sv_loss,
#     ylim = [1, 5],
#     ylabel = "cross entropy loss",
#     # filename = "results/supervised/vertex-4-1-sv-loss.png",
# )

num_trajectories = 500
nflip_range = 1:5:42
nn_ret = [returns_versus_nflips(policy, nref, nf, num_trajectories) for nf in nflip_range]
# gd_ret = [returns_versus_nflips(nref, nf, num_trajectories) for nf in nflip_range]
normalized_nflips = nflip_range ./ num_actions
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1])
plot_returns(normalized_nflips, nn_ret, gd_ret = gd_ret, ylim = [0.75,1], filename = "results/supervised/vertex-4-4-4-1-rl-vs-gd-10000.png")

# ret = PG.average_normalized_returns(env, policy, num_trajectories)
# gret = GreedyPolicy.average_normalized_returns(env, num_trajectories)