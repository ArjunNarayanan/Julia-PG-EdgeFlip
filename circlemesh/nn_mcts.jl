using EdgeFlip
include("../MCTS_utilities.jl")
include("../policy_and_value_network.jl")
include("circlemesh.jl")

TS = MCTS
PV = PolicyAndValueNetwork
EF = EdgeFlip

element_size = 0.2
env = circle_ordered_game_env(element_size, maxflipfactor = 1)

# using BSON
# policy = PV.PVNet(3, 8)
BSON.@load "results/models/MCTS/8L.bson" policy

exploration_factor = 10
maxiter = 50
temperature = 1
discount = 1.0
tree_settings = TS.TreeSettings(exploration_factor, maxiter, temperature, discount)

l2_coeff = 1e-3
memory_size = 500
num_epochs = 200
batch_size = 50
num_iter = 10
threshold = 0.6
num_samples = 500

iter_settings = TS.PolicyIterationSettings(
    discount,
    l2_coeff,
    memory_size,
    batch_size,
    num_epochs,
    num_iter,
    threshold,
    num_samples,
)

TS.reset!(env)
maxreturn = env.score - env.optimum_score