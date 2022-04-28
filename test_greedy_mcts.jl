include("GreedyMCTS_utilities.jl")

function GTS.number_of_actions(env::EdgeFlip.GameEnv)
    return EdgeFlip.number_of_actions(env)
end

env = EdgeFlip.GameEnv(1,10,maxflips=10)

exploration_factor = 1
maxiter = 500
temperature = 1
discount = 1
tree_settings = GTS.TreeSettings(exploration_factor, maxiter, temperature, discount)

value = estimate_value(env)
root = GTS.Node(env, value)

GTS.search!(root, env, tree_settings)