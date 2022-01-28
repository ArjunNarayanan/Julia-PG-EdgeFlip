using Flux
using EdgeFlip
import EdgeFlip: state, step!, is_terminated, reward, reset!
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
    return fig
end


function render_policy(env,policy)
    fig,ax = subplots()
    done = is_terminated(env)
    EdgeFlip.plot!(ax,env.mesh,d0=env.d0)
    while !done
        sleep(1)
        action = env |> state |> policy |> softmax |> Categorical |> rand
        step!(env,action)
        EdgeFlip.averagesmoothing!(env.mesh,3)
        EdgeFlip.plot!(ax,env.mesh,d0=env.d0)
        done = is_terminated(env)        
    end
end



learning_rate = 0.1
batch_size = 10
num_epochs = 1000

using Random
Random.seed!(10)
env = EdgeFlip.GameEnv(1, 5)

policy = Policy(Dense(4,1))

loss_history,return_history = run_training_loop(env,policy,batch_size,num_epochs,learning_rate)

using PyPlot
optimum = env.score
fig = plot_history(return_history,optimum=optimum)

pygui(true)
reset!(env)
render_policy(env,policy)


# reset!(env)
# fig,ax = EdgeFlip.render(env)
# action = env |> state |> policy |> softmax |> Categorical |> rand
# step!(env,action)
# fig,ax = EdgeFlip.render(env)
# r = reward(env)
# action = env |> state |> policy |> softmax |> Categorical |> rand
# step!(env,action)
# r = reward(env)