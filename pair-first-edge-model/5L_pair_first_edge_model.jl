using Printf
using Flux
using EdgeFlip
include("pair-first-policy.jl")
# include("plot.jl")

PG = EdgePolicyGradient

function PG.reset!(
    env::EdgeFlip.OrderedGameEnv;
    nflips = env.num_initial_flips,
    maxflips = env.maxflips,
)
    EdgeFlip.reset!(env, nflips = nflips, maxflips = maxflips)
end

function PG.state(env::EdgeFlip.OrderedGameEnv)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    na = length(epairs)
    idx = findall(epairs .== 0)
    epairs[idx] .= na + 1

    return ets, econn, epairs
end

function PG.step!(env::EdgeFlip.OrderedGameEnv, action)
    triangle, vertex = action
    EdgeFlip.step!(env, triangle, vertex)
end

function PG.is_terminated(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.is_terminated(env)
end

function PG.reward(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.reward(env)
end

function PG.score(env::EdgeFlip.OrderedGameEnv)
    return EdgeFlip.score(env)
end

function returns_versus_nflips(policy, nref, nflips, num_trajectories; maxstepfactor = 1.0)
    maxflips = ceil(Int, maxstepfactor * nflips)
    env = EdgeFlip.OrderedGameEnv(nref, nflips, maxflips = maxflips)
    avg = PG.average_normalized_returns(env, policy, num_trajectories)
    @printf "NFLIPS = %d \t MAXFLIPS = %d \t RET = %1.3f\n" env.num_initial_flips env.maxflips avg
    return avg
end


env = EdgeFlip.OrderedGameEnv(1, 10, maxflips = 10)
num_actions = EdgeFlip.number_of_actions(env)
policy = PairPolicy(3, 16)

PG.reset!(env)
ep, econn, epairs = PG.state(env)

num_trajectories = 500
batch_size = 100
num_epochs = 10000
learning_rate = 1e-2
decay = 0.7
decay_step = 500
clip = 5e-5
discount = 1

optimizer =
    Flux.Optimiser(ExpDecay(learning_rate, decay, decay_step, clip), ADAM(learning_rate))
# # optimizer = ADAM(5e-6)

PG.run_training_loop(env, policy, optimizer, batch_size, discount, num_epochs)

returns_versus_nflips(policy, 1, 10, 500)