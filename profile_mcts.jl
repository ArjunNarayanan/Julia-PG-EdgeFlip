include("MCTS_utilities.jl")
include("policy_and_value_network.jl")
using BenchmarkTools
using Profile

TS = MCTS
PV = PolicyAndValueNetwork

function run_evaluations(policy, ets, econn, epairs, remflips)
    for i = 1:1000
        l, v = PV.eval_single(policy, ets, econn, epairs, remflips)
    end
end

Cpuct = 1
temperature = 25
l2_coeff = 1e-3
discount = 1.0
maxtime = 1e-1
batch_size = 50
num_epochs = 50

nref = 1
nflips = 10
env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = nflips)
policy = PV.PVNet(3, 16)

TS.reset!(env)
p,v = TS.action_probabilities_and_value(policy, TS.state(env))
root = TS.Node(p, v, false)

ets, econn, epairs, remflips = TS.state(env)
Profile.clear()
@profile run_evaluations(policy, ets, econn, epairs, remflips)