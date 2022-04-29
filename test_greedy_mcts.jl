include("GreedyMCTS_utilities.jl")

function GTS.number_of_actions(env::EF.GameEnv)
    return EF.number_of_actions(env)
end

env = EF.GameEnv(1,10,maxflips=10)

exploration_factor = 0.05
maxiter = 500
temperature = 0.001
discount = 1
tree_settings = GTS.TreeSettings(exploration_factor, maxiter, temperature, discount)

mcts_ret, mcts_dev = average_normalized_tree_returns(env, tree_settings, 50)

# gd_ret, gd_dev = GP.average_normalized_returns(env, 500)