using TriMeshGame
include("TriMeshGame_PPO_utilities.jl")
using MeshPlotter

TM = TriMeshGame
MP = MeshPlotter

function hex_mesh()
    d = range(0,stop=2pi,length=7)[1:end-1]
    p = [cos.(d) sin.(d)]
    t = [1 2 6
         2 5 6
         2 3 5
         3 4 5]
    mesh = TM.Mesh(p, t)
    return mesh
end

function initialize_environment(nref, num_random_flips, max_actions)
    mesh = TM.circlemesh(nref)
    wrapper = GameEnvWrapper(mesh, mesh.d, num_random_flips, max_actions)
    return wrapper
end

function initialize_hex_environment(max_actions)
    mesh = hex_mesh()
    d0 = [3,3,3,3,3,3]
    wrapper = GameEnvWrapper(mesh, d0, 0, max_actions)
    return wrapper
end


function active_mesh(mesh)
    p = mesh.p
    t = mesh.t[mesh.active_triangle, :]
    return TM.Mesh(p, t)
end

function evaluator(policy, wrapper; num_trajectories = 100)
    ret, dev = average_normalized_returns(wrapper, policy, num_trajectories)
    return ret, dev
end

nref = 1
num_random_flips = 10
max_actions = 20
discount = 0.95
epsilon = 0.1
batch_size = 200
episodes_per_iteration = 1000
num_epochs = 5
num_iter = 50

wrapper = initialize_environment(nref, num_random_flips, max_actions)
# wrapper = initialize_hex_environment(3)
policy = SplitPolicy.Policy(24, 32, 3)
optimizer = ADAM(1e-3)

wrapper = initialize_environment(nref, num_random_flips, max_actions)
nt0 = size(wrapper.env.mesh.t,1)

ret = single_trajectory_normalized_return(wrapper, policy)
nt1 = size(wrapper.env.mesh.t,1)



# PPO.ppo_iterate!(
#     policy,
#     wrapper,
#     optimizer,
#     episodes_per_iteration,
#     discount,
#     epsilon,
#     batch_size,
#     num_epochs,
#     num_iter,
#     evaluator,
# )