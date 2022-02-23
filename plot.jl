using PyPlot
pygui(true)
using MeshPlotter

function plot_history(epochs, history; optimum = nothing, opt_label = "", title = "", filename = "", ylim = [0.,1.2])
    fig, ax = subplots()
    ax.plot(epochs, history)
    if !isnothing(optimum)
        ax.plot(epochs, optimum * ones(length(epochs)), "--", label = opt_label)
    end
    ax.set_title(title)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Avg Return")
    ax.set_ylim(ylim)
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
        MeshPlotter.plot!(ax, env.mesh, d0 = env.d0)

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
    MeshPlotter.plot!(ax, env.mesh, d0 = env.d0)

    if length(filename) > 0
        savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
        fig.savefig(savepath)
    end
end

function render_policy(env; pause = 0.5, maxsteps = 20, filename = "", figsize = 10)
    fig, ax = subplots(figsize = (figsize, figsize))
    done = is_terminated(env)
    MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)
    counter = 1
    while !done && counter < maxsteps
        EdgeFlip.averagesmoothing!(env.mesh, 3)
        MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)

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
    MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)

    if length(filename) > 0
        savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
        fig.savefig(savepath)
    end
end

function plot_returns(nflips,ret,dev;filename="",ylim=(0.75,1.0))
    fig,ax = subplots()
    ax.plot(nflips,ret)
    ax.fill_between(nflips,ret+dev,ret-dev,alpha=0.2,facecolor="blue")
    ax.set_xlabel("Normalized number of random initial flips")
    ax.set_ylabel("Normalized returns")
    ax.set_title("Normalized returns vs initial flips for greedy algorithm")
    ax.set_ylim(ylim)
    if length(filename) > 0
        fig.savefig(filename)
    end
    return fig
end
