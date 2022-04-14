include("MCTS.jl")

using OrderedCollections

TS = MCTS

p = 1 ./ (fill(5,5))
root = TS.Node(p)

s = TS.PUCT_score(root.prior_probabilities, root.visit_count, root.mean_action_values, 1)
idx = TS.select_action(root, 1)