using Flux
using Distributions: Categorical
using Printf
using PyPlot

include("template_game_env.jl")
include("policy_gradient.jl")

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

function expected_reward(policy)
    states = all_states()
    probs = softmax(policy(states), dims = 1)
    fr = flip_reward(states)
    avg_policy_return = Flux.mean(probs[2,:]' .* fr)
    return avg_policy_return
end


env = TemplateEnv()
policy = Dense(4, 2)

lr = 0.5
batch_size = 100
num_epochs = 1000

θ = params(policy)

loss_history, return_history =
    run_training_loop(env, policy, batch_size, num_epochs, lr, 1, 100)


greedy_optimum = 0.24691358024691357
fig = plot_history(
    return_history,
    optimum = greedy_optimum,
    # filename = "results\\template-greedy-reward-history.png",
)

performance = expected_reward(policy)

w,b = params(policy)

states = all_states()
probs = softmax(policy(states),dims=1)

prob_diff = abs.(probs[1,:] - probs[2,:])

sortidx = sortperm(prob_diff)
prob_diff = prob_diff[sortidx]

ambiguous_states = states[:,sortidx[1:10]]
ambiguous_probs = prob_diff[1:10]
fr = flip_reward(ambiguous_states)