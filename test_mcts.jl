include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
using MeshPlotter

function evaluate_model(policy, env; num_trajectories = 500)
    ret = returns_versus_nflips(policy, env, num_trajectories)
    return ret
end


TS = MCTS
PV = PolicyAndValueNetwork
MP = MeshPlotter

Cpuct = 1
temperature = 25
l2_coeff = 1e-3
discount = 1.0
maxtime = 1e-1
batch_size = 50
num_epochs = 50


# nref = 1
# nflips = 10
# env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = nflips)
# policy = PV.PVNet(3, 16)
# na = TS.number_of_actions(root)

# TS.reset!(env)
# total_score = env.score

# p, v = TS.action_probabilities_and_value(policy, TS.state(env))
# root = TS.Node(p, v, TS.is_terminal(env))
# TS.search!(root, env, policy, Cpuct, discount, maxtime)
# ap = TS.mcts_action_probabilities(root.visit_count, na, temperature)
# action = rand(Categorical(ap))
# TS.step!(env, action)
# env.reward

optimizer = ADAM(2e-3)
TS.train!(policy, env, optimizer, Cpuct, discount, maxtime, temperature, batch_size, l2_coeff, num_epochs, evaluate_model)