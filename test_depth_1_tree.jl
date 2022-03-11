using EdgeFlip
include("tree_search.jl")
include("greedy_policy.jl")

TS = TreeSearch
GP = GreedyPolicy

tree_depth = 1
env = EdgeFlip.GameEnv(1,10)

root = TS.Node()
TS.grow_at!(root, env, tree_depth)
TS.collect_returns!(root)

ta1 = TS.actions(root)[1]
ga = GP.greedy_action(env)

ta = TS.actions(root,TS.best_child(root))
TS.step_best_trajectory!(env, root)

root = TS.Node()
TS.grow_at!(root, env, tree_depth)

ta = TS.actions(root)
ga = GP.greedy_actions(env)