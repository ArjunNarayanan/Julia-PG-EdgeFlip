using PyPlot
pygui(true)
using MeshPlotter
using Distributions: Categorical

function plot_history(
    epochs,
    history;
    optimum = nothing,
    opt_label = "",
    title = "",
    filename = "",
    ylim = [0.0, 1.2],
    ylabel = "Avg Return"
)
    fig, ax = subplots()
    ax.plot(epochs, history)
    if !isnothing(optimum)
        ax.plot(epochs, optimum * ones(length(epochs)), "--", label = opt_label)
    end
    ax.set_title(title)
    ax.grid()
    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    if length(filename) > 0
        fig.savefig(filename)
    end
    fig.tight_layout()
    return fig
end

function render_policy(env, policy; pause = 0.5, filename = "", figsize = 10)
    fig, ax = subplots(figsize = (figsize, figsize))
    done = EdgeFlip.is_terminated(env)
    MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)
    while !done
        EdgeFlip.averagesmoothing!(env.mesh, 3)
        MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)

        sleep(pause)
        logits = policy(state(env))
        probs = Categorical(vec(softmax(logits, dims = 2)))
        action = rand(probs)
        EdgeFlip.step!(env, action)
        done = EdgeFlip.is_terminated(env)

        if length(filename) > 0
            savepath = filename * "-" * lpad(counter, 2, "0") * ".png"
            fig.savefig(savepath)
        end

    end
    EdgeFlip.averagesmoothing!(env.mesh, 3)
    MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)

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

function plot_returns(nflips, ret, dev; filename = "", ylim = (0.75, 1.0))
    fig, ax = subplots()
    ax.plot(nflips, ret)
    ax.fill_between(nflips, ret + dev, ret - dev, alpha = 0.2, facecolor = "blue")
    ax.set_xlabel("Normalized number of random initial flips")
    ax.set_ylabel("Normalized returns")
    ax.set_title("Normalized returns vs initial flips for greedy algorithm")
    ax.set_ylim(ylim)
    if length(filename) > 0
        fig.savefig(filename)
    end
    return fig
end

function plot_returns(nflips, ret; gd_ret = [], filename = "", ylim = (0.75, 1.0), title = "")
    fig, ax = subplots()
    ax.plot(nflips, ret, label = "policy")
    if length(gd_ret) > 0
       ax.plot(nflips, gd_ret, label = "greedy")
    end
    ax.legend()
    ax.set_xlabel("Normalized number of random initial flips")
    ax.set_ylabel("Normalized returns")
    ax.set_title(title)
    ax.set_ylim(ylim)
    if length(filename) > 0
        fig.savefig(filename)
    end
    return fig
end

function plot_learning_curve!(ax, epochs, returns)
    ax.plot(epochs, returns, alpha = 0.6, color = "red")
end

function plot_learning_curves(data;)
    fig,ax = subplots()
    for d in data
        plot_learning_curve!(ax, d[:,1], d[:,2])
    end
    ax.grid()
    # ax.set_ylim(ylim)
    ax.set_xlabel("epochs")
    ax.set_ylabel("returns")
    return fig, ax
end