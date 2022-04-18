using Test
using EdgeFlip
using Distributions: Categorical

include("MCTS_utilities.jl")
include("policy_and_value_network.jl")

TS = MCTS
PV = PolicyAndValueNetwork

nref = 0
env = EdgeFlip.OrderedGameEnv(nref, 0)
policy = PV.PVNet(3, 16)

Cpuct = 1.0
temperature = 1.0
discount = 1.0
maxtime = 0.1

TS.reset!(env)
opt = env.score - env.optimum_score
batch_data = TS.BatchData(StateData())

p, v = TS.action_probabilities_and_value(policy, TS.state(env))
terminal = TS.is_terminal(env)
root = TS.Node(p, terminal)

old_score = env.score
TS.search!(root, env, policy, Cpuct, discount, maxtime)
new_score = env.score

c = root.children[12]

@test old_score == new_score

# na = TS.number_of_actions(root)
# ap = TS.mcts_action_probabilities(root.visit_count, na, temperature)
# action = rand(Categorical(ap))

# TS.step!(env,action)
# r = TS.reward(env)
# c = TS.child(root, action)
# t = TS.is_terminal(c)

# test_reward = (old_score - env.score)/old_score



# s2 = TS.state(env)

# TS.collect_sample_trajectory!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)



# TS.step_mcts!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)

# root = TS.search(env, policy, Cpuct, discount, maxtime)
# ap = TS.mcts_action_probabilities(root.visit_count, 72, 1.0)



# p = Categorical(TS.mcts_action_probabilities(root.visit_count, 72, temperature))


# n2, a2 = TS.select(root, env, 1)

# p, v = TS.action_probabilities_and_value(policy, TS.state(env))
# root = TS.Node(p)
# node, action = TS.select(root, env, 1)
# child, val = TS.expand(node, action, env, policy)
# root = TS.backup(child, val, 1.0, env)

# node, action = TS.select(root, env, 1)
# child, val = TS.expand(node, action, env, policy)