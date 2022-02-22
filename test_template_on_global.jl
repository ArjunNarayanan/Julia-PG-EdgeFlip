using Flux
using Distributions: Categorical
using EdgeFlip
# import EdgeFlip: state, step!, is_terminated, reward, reset!
using Printf
include("global_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")

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
    return vec(p.model(s))
end

Flux.@functor Policy

nref = 1
nflips = 8
maxflips = ceil(Int, 1.2nflips)
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)


learning_rate = 0.1
batch_size = 32
num_epochs = 1000
num_trajectories = 100


policy = Policy(Dense(4, 1))
# policy = Policy(Chain(Dense(4, 4, relu), Dense(4, 1)))



epoch_history, return_history = PolicyGradient.run_training_loop(
    env,
    policy,
    batch_size,
    num_epochs,
    learning_rate,
    num_trajectories,
    estimate_every = 100,
)


num_test_trajectories = 1000
nn_avg =
    PolicyGradient.average_normalized_returns(env, policy, num_test_trajectories)
greedy_avg = GreedyPolicy.average_normalized_returns(env, num_test_trajectories)
@printf "NN MEAN : %2.3f\n" nn_avg
@printf "GD MEAN : %2.3f\n" greedy_avg


# reset!(env)
# trial_num = 1
# gd_env = deepcopy(env)

# filename = "results/greedy-vs-nn-anim/nref" * string(nref) * "/nn/nn" * string(trial_num)
# render_policy(env, policy, filename = filename, figsize = 7, maxsteps = maxsteps)

# filename = "results/greedy-vs-nn-anim/nref" * string(nref) * "/gd/gd" * string(trial_num)
# render_policy(gd_env, figsize = 7, maxsteps = maxsteps)
