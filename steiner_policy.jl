using EdgeFlip
using Statistics
using PyPlot
using MeshPlotter
using CSV, DataFrames
pygui(true)

function single_trajectory_return(env)
    ep_returns = []
    done = EdgeFlip.is_terminated(env)
    if done
        return 0.0
    else
        while !done
            action = EdgeFlip.greedy_action(env)
            EdgeFlip.step!(env, action)
            push!(ep_returns, EdgeFlip.reward(env))
            done = EdgeFlip.is_terminated(env)
        end
        return sum(ep_returns)
    end
end

function single_trajectory_normalized_return(env)
    maxscore = EdgeFlip.score(env)
    if maxscore == 0
        return 0.0
    else
        ret = single_trajectory_return(env)
        return ret / maxscore
    end
end

function average_normalized_returns(nref, nflips, maxflips, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        # println("step = $idx")
        env = EdgeFlip.SteinerGameEnv(nref, nflips, maxflips = maxflips)
        ret[idx] = single_trajectory_normalized_return(env)
    end
    return mean(ret), std(ret)
end

function render_policy(env; pause = 0.5, filename = "", figsize = 10)
    fig, ax = subplots(figsize = (figsize, figsize))
    done = EdgeFlip.is_terminated(env)
    MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)
    counter = 1
    while !done
        EdgeFlip.averagesmoothing!(env.mesh, 3)
        MeshPlotter.plot_mesh!(ax, env.mesh, d0 = env.d0)

        sleep(pause)
        action = EdgeFlip.greedy_action(env)
        EdgeFlip.step!(env, action)
        done = EdgeFlip.is_terminated(env)

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

function returns_vs_nflips(nflips, num_trajectories; maxstepfactor = 1.2)
    avg = zeros(length(nflips))
    dev = zeros(length(nflips))
    for (idx, nf) in enumerate(nflips)
        maxflips = ceil(Int, maxstepfactor * nf)
        a, d =
        average_normalized_returns(nref, nf, maxflips, num_trajectories)
        avg[idx], dev[idx] = a, d
        println("nflips = $nf \t avg = $a \t dev = $d")
    end
    return avg, dev
end

nref = 1
nflips = 1:30
# nflips = 15
# maxflips = nflips

# env = EdgeFlip.SteinerGameEnv(nref,15,maxflips=maxflips)
# ret = single_trajectory_normalized_return(env)

# render_policy(env)

# avg, dev = returns_vs_nflips(nflips, 50, maxstepfactor = 1.2)

# mesh = EdgeFlip.generate_mesh(nref)
# normalized_nflips = nflips ./ EdgeFlip.number_of_edges(mesh)

# df = DataFrame(:nflips => normalized_nflips, :avg => avg, :dev => dev)
# filename = "results/steiner-greedy-vs-nflips/nref-1.csv"
# CSV.write(filename,df)

# fig,ax = PyPlot.subplots()
# ax.scatter(normalized_nflips, avg, label = "greedy + steiner")
# ax.scatter(normalized_nflips, returns, label = "greedy")
# ax.grid()
# ax.legend()
# ax.set_ylim(0.8,1.1)
# ax.set_xlabel("normalized number of initial flips")
# ax.set_ylabel("normalized average returns")
# fig.savefig("results/steiner-greedy-vs-nflips/steiner-vs-greedy.png")
