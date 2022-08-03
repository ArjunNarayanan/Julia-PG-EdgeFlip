# using StatsBase
using Statistics
using Printf
using EdgeFlip
using MeshPlotter
include("greedy_policy.jl")
include("tree_search.jl")
include("plot.jl")

TS = TreeSearch

function returns_vs_nflips(
    nref,
    nflips,
    tree_depth,
    max_branching_factor,
    num_trajectories;
    maxstepfactor = 1.0,
)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = TS.average_normalized_returns(
        env,
        tree_depth,
        max_branching_factor,
        num_trajectories,
    )
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" nflips maxflips avg
    return avg
end

function returns_vs_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.0)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips, maxflips = maxflips)
    avg = GreedyPolicy.average_normalized_returns(env, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" nflips maxflips avg
    return avg
end

nref = 1
num_actions = 42
num_trajectories = 500
nflip_range = 1:5:42

# gret = [returns_vs_nflips(nref, nf, num_trajectories) for nf in nflip_range]
# tree_depth = 20
# max_branching_factor = 3
# tret = [returns_vs_nflips(nref, nf, tree_depth, max_branching_factor, num_trajectories) for nf in nflip_range]

# normalized_nflips = nflip_range ./ num_actions
# plot_returns(
#     normalized_nflips,
#     tret,
#     gd_ret = gret,
#     ylim = [0.75, 1.0],
#     label = "tree",
#     # filename = "results/tree-search/tree-vs-nflips-d-4.png",
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