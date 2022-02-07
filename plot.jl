using PyPlot
pygui(true)

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

function render_policy(env, policy; pause = 0.5, maxsteps = 20, filename = "", figsize = 10)
    fig, ax = subplots(figsize = (figsize, figsize))
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
            savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
            fig.savefig(savepath)
        end

        counter += 1
    end
    EdgeFlip.averagesmoothing!(env.mesh, 3)
    EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)

    if length(filename) > 0
        savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
        fig.savefig(savepath)
    end
end

function render_policy(env; pause = 0.5, maxsteps = 20, filename = "", figsize = 10)
    fig, ax = subplots(figsize = (figsize, figsize))
    done = is_terminated(env)
    EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)
    counter = 1
    while !done && counter < maxsteps
        EdgeFlip.averagesmoothing!(env.mesh, 3)
        EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)

        sleep(pause)
        action = GreedyPolicy.greedy_action(env)
        step!(env, action)
        done = is_terminated(env)

        if length(filename) > 0
            savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
            fig.savefig(savepath)
        end

        counter += 1
    end
    EdgeFlip.averagesmoothing!(env.mesh, 3)
    EdgeFlip.plot!(ax, env.mesh, d0 = env.d0)

    if length(filename) > 0
        savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
        fig.savefig(savepath)
    end
end
