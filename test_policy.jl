using Flux
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

function expected_reward(states, policy)
    probs = softmax(policy(states), dims = 1)
    fr = flip_reward(states)

end


env = TemplateEnv()
policy = Dense(4, 2)

lr = 1.0
batch_size = 32
num_epochs = 100

θ = params(policy)

loss_history, return_history =
    run_training_loop(env, policy, batch_size, 100, lr)


greedy_optimum = 0.24691358024691357
# fig = plot_history(
#     return_history,
#     optimum = greedy_optimum,
#     filename = "results\\template-greedy-reward-history.png",
# )

states = all_states()
probs = softmax(policy(states),dims=1)
fr = flip_reward(states)
avg_policy_return = Flux.mean(probs[2,:]' .* fr)


s = [-1,-1,1,1]
probs = softmax(policy(s))
println(probs)
