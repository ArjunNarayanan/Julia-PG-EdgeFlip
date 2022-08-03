using Printf
include("GreedyMCTS_utilities.jl")

function returns_vs_nflips(
    nref,
    nflips,
    tree_settings,
    num_trajectories;
    maxstepfactor = 1.,
)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EF.GameEnv(nref, nflips, maxflips = maxflips)
    avg, dev = average_normalized_tree_returns(env, tree_settings, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f \t DEV = %1.3f\n" env.num_initial_flips env.maxflips avg dev
    return avg, dev
end

exploration_factor = 0.1
maxiter = 200
temperature = 0.001
discount = 1
tree_settings = GTS.TreeSettings(exploration_factor, maxiter, temperature, discount)

avg, dev = returns_vs_nflips(1, 15, tree_settings, 50)

# nflips_range = 5:5:42
# stats = [returns_vs_nflips(1, nflips, tree_settings, 50) for nflips in nflips_range]

# gd_ret, gd_dev = GP.average_normalized_returns(env, 500)
