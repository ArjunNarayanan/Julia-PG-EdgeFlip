include("MCTS_utilities.jl")
include("policy_and_value_network.jl")

function evaluate_model(policy; num_trajectories = 500)
    nref = 0
    nflips = 1
    ret = returns_versus_nflips(policy, nref, nflips, num_trajectories)
    return ret
end

TS = MCTS
PV = PolicyAndValueNetwork

Cpuct = 1.0
temperature = 1e0
l2_coeff = 1e-3
discount = 1.0
maxtime = 1e-2
batch_size = 50
num_epochs = 100


# nref = 0
# nflips = 1
# env = EdgeFlip.OrderedGameEnv(nref, nflips)
# policy = PV.PVNet(3, 16)
# na = EdgeFlip.number_of_actions(env)

TS.reset!(env, nflips = 1)
# p, v = TS.action_probabilities_and_value(policy, TS.state(env))
# root = TS.Node(p,v,TS.is_terminal(env))
# TS.search!(root, env, policy, Cpuct, discount, maxtime)
# ap = TS.mcts_action_probabilities(root.visit_count, 18, 1e2)

optimizer = ADAM(0.002)
TS.train!(policy, env, optimizer, Cpuct, discount, maxtime, temperature, batch_size, l2_coeff, num_epochs, evaluate_model)

# TS.step_epoch!(policy, env, optimizer, Cpuct, discount, maxtime, temperature, batch_size, l2_coeff)

# TS.reset!(env, nflips = 1)
# p, v = TS.action_probabilities_and_value(policy, TS.state(env))
# root = TS.Node(p,v,TS.is_terminal(env))
# root = TS.step_mcts!(data, root, env, policy, Cpuct, discount, maxtime, temperature)
# terminal = TS.is_terminal(root)
