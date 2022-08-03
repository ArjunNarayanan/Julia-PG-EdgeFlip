using TriMeshGame
include("random_polygon_generator.jl")
include("RandDegree_PPO_utilities.jl")
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


degree_range = [2,3,4]

dd = rand(degree_range,5)
angles = valence2degrees.(dd)
sum_angles = sum(angles)
remaining_angle = 720 - sum_angles

max_actions = 10

discount = 0.95
epsilon = 0.1
batch_size = 200
episodes_per_iteration = 1000
num_epochs = 5
num_iter = 50

wrapper = initialize_hex_environment(degree_range,20)
# policy = SplitPolicy.Policy(24, 32, 5)
optimizer = ADAM(1e-3)

# MP.plot_mesh(wrapper.env.mesh, d0=wrapper.desired_degree)[1]

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

# using BSON: @save
# @save "results\\random-degree\\rand-degree-policy.bson" policy

# using BSON: @load
# @load "results\\random-degree\\rand-degree-policy.bson" policy

# evaluator(policy, wrapper)

# average_number_of_splits(wrapper,policy,100)

# PPO.reset!(wrapper)
# MP.plot_mesh(wrapper.env.mesh,d0=wrapper.desired_degree)[1]

# w0 = deepcopy(wrapper)

# MP.plot_mesh(w0.env.mesh,d0=w0.desired_degree)[1]

# single_trajectory_normalized_return(wrapper,policy)

# act_mesh = active_mesh(wrapper.env.mesh)
# MP.plot_mesh(act_mesh,d0=wrapper.desired_degree)[1]

# valid_splits, invalid_splits = average_number_of_splits(wrapper, policy, 100)
