using Flux
using Distributions: Categorical
using EdgeFlip
import EdgeFlip: state, step!, is_terminated, reward, reset!
using Printf
include("global_policy_gradient.jl")



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

function global_score(env)
    return sum(abs.(env.vertex_score))
end

function plot_history(history; optimum = nothing, title = "", filename = "")
    fig, ax = subplots()
    ax.plot(history, label = "avg reward")
    if !isnothing(optimum)
        ax.plot(optimum * ones(length(history)), "--", label = "optimum")
    end
    ax.set_title(title)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Epochs")
    if length(filename) > 0
        fig.savefig(filename)
    end
    fig.tight_layout()
    return fig
end


function render_policy(env, policy; pause = 0.5, maxsteps = 20, filename = "")
    fig, ax = subplots()
    done = is_terminated(env)
    EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)
    counter = 1
    while !done && counter < maxsteps
        EdgeFlip.averagesmoothing!(env.mesh, 3)
        EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)

        sleep(pause)
        action = env |> state |> policy |> softmax |> Categorical |> rand
        step!(env, action)
        done = is_terminated(env)

        if length(filename) > 0
            savepath = filename * "-" * lpad(counter,2,"0") * ".png"
            fig.savefig(savepath)
        end

        counter += 1
    end
    EdgeFlip.averagesmoothing!(env.mesh, 3)
    EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)

    if length(filename) > 0
        savepath = filename * "-" * lpad(counter,2,"0") * ".png"
        fig.savefig(savepath)
    end
end


learning_rate = 0.1
batch_size = 100
num_epochs = 1000
maxsteps = 20
num_trajectories = 100

env = EdgeFlip.GameEnv(2, maxsteps, fixed_reset = true)
EdgeFlip.render(env)[1]

policy = Policy(Chain(Dense(4, 4, relu),Dense(4,1)))
loss_history, return_history = PolicyGradient.run_training_loop(env, policy, batch_size, num_epochs, learning_rate, maxsteps, num_trajectories, estimate_every=100)


using PyPlot
reset!(env)
optimum = env.score
# ret = average_returns(env, policy)
# fig = plot_history(return_history,optimum=optimum,title="Average reward vs Epochs")

pygui(true)
reset!(env)
render_policy(env, policy)
