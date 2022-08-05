using TriMeshGame
include("PPO/RandPoly_PPO_utilities.jl")
using MeshPlotter

TM = TriMeshGame
MP = MeshPlotter

function active_mesh(mesh)
    p = mesh.p
    t = mesh.t[mesh.active_triangle, :]
    return TM.Mesh(p, t)
end

function evaluator(policy, wrapper; num_trajectories = 100)
    ret, dev = average_normalized_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

function returns_vs_maxflips(policy, polyorder, max_actions; threshold = 0.1)
    wrapper = GameEnvWrapper(polyorder, threshold, max_actions)
    ret, dev = evaluator(policy, wrapper; num_trajectories = 1000)
    return ret, dev
end

function best_returns_vs_maxflips(policy, polyorder, max_actions; threshold = 0.1)
    wrapper = GameEnvWrapper(polyorder, threshold, max_actions)
    ret, dev = average_best_normalized_returns(wrapper, policy, 20, 500)
    return ret, dev
end

polyorder = 10
threshold = 0.2
max_actions = 20

discount = 0.90
epsilon = 0.1
batch_size = 20
episodes_per_iteration = 200
num_epochs = 10
num_iter = 1000

# wrapper = GameEnvWrapper(polyorder, threshold, max_actions)
# policy = SplitPolicy.Policy(12, 32, 5)
# optimizer = ADAM(1e-3)



PPO.ppo_iterate!(
    policy,
    wrapper,
    optimizer,
    episodes_per_iteration,
    discount,
    epsilon,
    batch_size,
    num_epochs,
    num_iter,
    evaluator,
)



# function plot_returns(maxflips, ret, dev; filename = "", ylim = (0.0, 1.0))
#     fig, ax = subplots()
#     ax.plot(maxflips, ret)
#     ax.fill_between(maxflips, ret + dev/2, ret - dev/2, alpha = 0.2, facecolor = "blue")
#     ax.set_xlabel("Maximum allowed actions")
#     ax.set_ylabel("Normalized returns")
#     ax.set_ylim(ylim)
#     if length(filename) > 0
#         fig.savefig(filename)
#     end
#     return fig
# end


# using PyPlot

# polyorder = 6
# max_actions_range = 0:2:50
# stats = [best_returns_vs_maxflips(policy, polyorder, ma) for ma in max_actions_range]
# ret = [first(s) for s in stats]
# dev = [last(s) for s in stats]

# fig = plot_returns(max_actions_range, ret, dev)
# filename = "results/ppo-random-polygon/figures/best-performance-"*string(polyorder)*".png"
# fig = plot_returns(max_actions_range, ret, dev, filename = filename)


# using BSON: @save
# @save "results/ppo-random-polygon/poly10-vs-d-d0-policy.bson" policy

# using BSON: @load
# @load "results\\random-degree\\rand-degree-policy.bson" policy

# polyorder = 10
# wrapper = GameEnvWrapper(polyorder, 0.2, 30)

# counter = 0

# PPO.reset!(wrapper)

# fig, ax = MP.plot_mesh(wrapper.mesh0,d0=wrapper.desired_degree)
# fig
# counter += 1
# filename = "results/ppo-random-polygon/figures/initial-"*string(counter)*".png"
# fig.savefig(filename)

# initial_reset!(wrapper)
# s = best_policy_mesh!(wrapper, policy)

# act_mesh = active_mesh(wrapper.env.mesh)
# fig, ax = MP.plot_mesh(act_mesh,d0=wrapper.env.d0)
# fig
# filename = "results/ppo-random-polygon/figures/improved-"*string(counter)*".png"
# fig.savefig(filename)
# valid_splits, invalid_splits = average_number_of_splits(wrapper, policy, 100)
