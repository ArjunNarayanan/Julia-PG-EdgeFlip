using TriMeshGame
include("TriMeshGame_PPO_utilities.jl")
using MeshPlotter
using Distributions: Categorical

TM = TriMeshGame
MP = MeshPlotter

function initialize_environment(nref, num_random_flips, max_actions)
    mesh = TM.circlemesh(nref)
    d0 = copy(mesh.d)
    TM.random_flips!(mesh, num_random_flips)

    return TM.GameEnv(mesh,d0,max_actions)
end

function active_mesh(mesh)
    p = mesh.p
    t = mesh.t[mesh.active_triangle,:]
    return TM.Mesh(p,t)
end

nref = 1
num_random_flips = 10
max_actions = 20

env = initialize_environment(nref, num_random_flips, max_actions)
policy = SplitPolicy.Policy(24,64,5)

episode_data = PPO.EpisodeData(PPO.initialize_state_data(env))

PPO.collect_episode_data!(episode_data, env, policy)