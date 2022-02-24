using Flux
using Distributions: Categorical
using EdgeFlip
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

function get_learning_curve(discount, num_epochs)
    nref = 1
    nflips = 8
    maxflips = ceil(Int, 1.2nflips)
    env = EdgeFlip.GameEnv(nref, nflips, fixed_reset = false, maxflips = maxflips)

    # policy = Policy(Dense(4, 1))
    policy = Policy(Chain(Dense(4, 4, relu), Dense(4, 1)))
    # policy = Policy(Chain(Dense(4,8,relu), Dense(8,8,relu), Dense(8,1)))

    learning_rate = 0.1
    batch_size = 32
    num_trajectories = 100

    epoch_history, return_history = PolicyGradient.run_training_loop(
        env,
        policy,
        batch_size,
        discount,
        num_epochs,
        learning_rate,
        num_trajectories,
        estimate_every = 100,
    )

    return hcat(epoch_history, return_history)

end

using CSV, DataFrames

function write_learning_curve(data,filename)
    df = DataFrame("epochs" => data[:,1], "returns" => data[:,2])
    CSV.write(filename,df)
end

function write_learning_curves(curves, foldername)
    numcurves = length(curves)
    for (idx,curve) in enumerate(curves)
        filename = foldername * "curve-"*string(idx)*".csv"
        write_learning_curve(curve, filename)
    end
end

num_epochs = 3000

discount = 0.8

# foldername = "results/discount-analysis/discount-"*string(discount)*"/"
curves = [get_learning_curve(discount, num_epochs) for k = 1:10]

# write_learning_curves(curves, foldername)