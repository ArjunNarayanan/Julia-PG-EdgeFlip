# using StatsBase
using Statistics
using Printf
using EdgeFlip
using MeshPlotter
include("greedy_policy.jl")
include("tree_search.jl")
include("plot.jl")

TS = TreeSearch

function single_trajectory_return(env, tree_depth)
    initial_score = EdgeFlip.score(env)
    done = EdgeFlip.is_terminated(env)
    if done
        return 0.0
    else
        while !done
            TS.step_tree_search!(env, tree_depth)
            done = EdgeFlip.is_terminated(env)
        end
        final_score = EdgeFlip.score(env)
        return initial_score - final_score
    end
end

function single_trajectory_normalized_return(env, tree_depth)
    maxscore = EdgeFlip.score(env)
    if maxscore == 0
        return 1.0
    else
        ret = single_trajectory_return(env, tree_depth)
        return ret / maxscore
    end
end

function normalized_returns(env, tree_depth, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        EdgeFlip.reset!(env)
        ret[idx] = single_trajectory_normalized_return(env, tree_depth)
    end
    return ret
end

function average_normalized_returns(env, tree_depth, num_trajectories)
    ret = normalized_returns(env, tree_depth, num_trajectories)
    return sum(ret) / length(ret)
end

function returns_vs_nflips(nref, nflips, tree_depth, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = average_normalized_returns(env, tree_depth, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" nflips maxflips avg
    return avg
end

function returns_vs_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" nflips maxflips avg
    return avg
end

tree_depth = 3
nref = 1
num_trajectories = 500
nflip_range = 1:5:42

num_actions = EdgeFlip.number_of_edges(EdgeFlip.generate_mesh(nref))

# tret = [returns_vs_nflips(nref, nf, tree_depth, num_trajectories) for nf in nflip_range]
# gret = [returns_vs_nflips(nref, nf, num_trajectories) for nf in nflip_range]

# normalized_nflips = nflip_range ./ num_actions
# plot_returns(
#     normalized_nflips,
#     tret,
#     gd_ret = gret,
#     ylim = [0.8, 1.0],
#     label = "tree",
#     filename = "results/tree-search/tree-vs-nflips-d-4.png",
# )

# nflips = 8
# maxflip_range = 1:0.1:3.0
# tret = [
#     returns_vs_nflips(nref, nflips, tree_depth, num_trajectories; maxstepfactor = mf) for
#     mf in maxflip_range
# ]
# gret = [
#     returns_vs_nflips(nref, 8, num_trajectories, maxstepfactor = mf) for mf in maxflip_range
# ]

# plot_returns(
#     maxflip_range,
#     tret,
#     gd_ret = gret,
#     ylim = [0.8, 1.0],
#     xlabel = "maxflips/nflips",
#     label = "tree",
#     # filename = "results/tree-search/tree-vs-maxflipratio.png",
# )