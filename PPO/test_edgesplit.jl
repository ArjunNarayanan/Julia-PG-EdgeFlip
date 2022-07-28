using TriMeshGame
include("TriMeshGame_PPO_utilities.jl")
using MeshPlotter

TM = TriMeshGame
MP = MeshPlotter

function initialize_environment(nref, num_random_flips, max_actions)
    mesh = TM.circlemesh(nref)
    wrapper = GameEnvWrapper(mesh, mesh.d, num_random_flips, max_actions)
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
batch_size = 10
episodes_per_iteration = 1000
num_epochs = 5
num_iter = 50

# wrapper = initialize_environment(nref, num_random_flips, max_actions)
# policy = SplitPolicy.Policy(24, 64, 5)
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

# rollouts = PPO.Rollouts()
# PPO.collect_rollouts!(rollouts, wrapper, policy, discount, episodes_per_iteration)

# PPO.ppo_train!(policy, optimizer, rollouts, epsilon, batch_size, num_epochs)

# state = PPO.batch_state(PPO.state_data.(minibatch))
# old_ap = PPO.batch_selected_action_probabilities(minibatch)
# adv = PPO.batch_advantage(minibatch)
# sel_actions = PPO.batch_selected_actions(minibatch)

# ap = PPO.batch_action_probabilities(policy, state)
# loss = PPO.ppo_loss(policy, state, sel_actions, old_ap, adv, 0.2)

# old_ap = PPO.batch_selected_action_probabilities(minibatch)
# advantage = PPO.batch_advantage(minibatch)
# sel_actions = PPO.batch_selected_actions(minibatch)