using EdgeFlip
using MeshPlotter
include("GreedyMCTS_utilities.jl")
include("circlemesh.jl")

GTS = GreedyMCTS

element_size = 0.4
env = circle_env(element_size, maxflipfactor = 2)
num_actions = EdgeFlip.number_of_actions(env)

exploration = 0.1
maxiter = 500
temperature = 0.001
discount = 1.
tree_settings = GTS.TreeSettings(exploration, maxiter, temperature, discount)

EdgeFlip.reset!(env)
maxreturn = env.score - env.optimum_score
minscore = EF.score(env)

ap = GTS.mcts_action_probabilities([0,1], 1)


# using Distributions: Categorical
# using Flux: softmax
# root = GTS.Node(env)
# # GTS.search!(root, env, tree_settings)
# ap = GTS.tree_action_probabilities!(root, env, tree_settings)
# action = rand(Categorical(ap))

# root = GTS.step_mcts!(root, env, tree_settings)
# minscore = min(minscore, EF.score(env))

# single_trajectory_normalized_tree_return(env, tree_settings)