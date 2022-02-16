using Flux
using Distributions: Categorical
using EdgeFlip
import EdgeFlip: state, step!, is_terminated, reward, reset!
using Printf
include("global_policy_gradient.jl")
include("greedy_policy.jl")
include("plot.jl")


struct Policy
    model::Any
    function Policy(model)
        new(model)
    end
end

function Flux.params(p::Policy)
    return params(p.model)
end

function (p::Policy)(s)
    return vec(p.model(s))
end

nref = 2
nflips = 20
env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false)


learning_rate = 0.1
batch_size = 32
num_epochs = 5000
maxsteps = ceil(Int, 1.2nflips)
num_trajectories = 100


policy = Policy(Dense(4, 1))
# policy = Policy(Chain(Dense(4, 4, relu), Dense(4, 1)))


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


num_test_trajectories = 1000
nn_avg, nn_dev =
    PolicyGradient.mean_and_std_returns(env, policy, maxsteps, num_test_trajectories)
greedy_avg, greedy_dev =
    GreedyPolicy.mean_and_std_returns(env, maxsteps, num_test_trajectories)
@printf "NN MEAN : %2.3f \t NN DEV : %2.3f\n" nn_avg nn_dev
@printf "GD MEAN : %2.3f \t GD DEV : %2.3f\n" greedy_avg greedy_dev


reset!(env)
# trial_num = 1
gd_env = deepcopy(env)

# filename = "results/greedy-vs-nn-anim/nref" * string(nref) * "/nn/nn" * string(trial_num)
# render_policy(env, policy, filename = filename, figsize = 7, maxsteps = maxsteps)

# filename = "results/greedy-vs-nn-anim/nref" * string(nref) * "/gd/gd" * string(trial_num)
render_policy(gd_env, figsize = 7, maxsteps = maxsteps)
