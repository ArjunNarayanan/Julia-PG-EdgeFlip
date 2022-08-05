using Flux
using Distributions: Categorical
using TriMeshGame
include("../utilities/random_polygon_generator.jl")
include("PPO.jl")
include("PPO_split_policy.jl")


TM = TriMeshGame

function random_polygon_mesh(polyorder, threshold)
    p = random_coordinates(polyorder, threshold=threshold)
    pclosed = [p' p[1,:]] 
    pout, t = polytrimesh([pclosed], holes=[], cmd="pQ")
    t = permutedims(t,(2,1))
    mesh = TM.Mesh(p, t)
    return mesh
end

mutable struct GameEnvWrapper
    mesh0::Any
    polyorder
    threshold
    desired_degree::Any
    max_actions::Any
    env::Any
    function GameEnvWrapper(polyorder, threshold, max_actions)
        mesh0 = random_polygon_mesh(polyorder, threshold)
        desired_degree = desired_valence.(polygon_interior_angles(mesh0.p))
        env = TM.GameEnv(mesh0, desired_degree, max_actions)
        new(mesh0, polyorder, threshold, desired_degree, max_actions, env)
    end
end

function Base.show(io::IO, wrapper::GameEnvWrapper)
    println(io, "Wrapped Environment:")
    show(io, wrapper.env)
end

function val_or_zero(vector, template)
    return [t == 0 ? 0 : vector[t] for t in template]
end

function val_or_missing(vector, template, missing_val)
    return [t == 0 ? missing_val : vector[t] for t in template]
end

function initial_reset!(wrapper::GameEnvWrapper)
    # wrapper.desired_degree = desired_valence.(polygon_interior_angles(wrapper.mesh0.p))
    wrapper.env = TM.GameEnv(wrapper.mesh0, wrapper.desired_degree, wrapper.max_actions)
end

function random_reset!(wrapper::GameEnvWrapper)
    wrapper.mesh0 = random_polygon_mesh(wrapper.polyorder, wrapper.threshold)
    wrapper.desired_degree = desired_valence.(polygon_interior_angles(wrapper.mesh0.p))
    wrapper.env = TM.GameEnv(wrapper.mesh0, wrapper.desired_degree, wrapper.max_actions)
end

function PPO.state(wrapper)
    env = wrapper.env

    # vs = val_or_zero(env.vertex_score, env.template)
    # vd = val_or_zero(env.mesh.d, env.template)
    # v0 = val_or_zero(env.d0, env.template)
    vs = val_or_missing(env.vertex_score, env.template, 10)

    return vs
    # return vcat(vs, vd, v0)
end

function PPO.reset!(wrapper)
    random_reset!(wrapper)
end

function PPO.is_terminal(wrapper)
    env = wrapper.env
    return env.is_terminated
end

function PPO.reward(wrapper)
    env = wrapper.env
    return env.reward
end

function index_to_action(index)
    triangle = div(index - 1, 6) + 1

    pos = rem(index - 1, 6)
    vertex = div(pos, 2) + 1
    type = rem(pos, 2) + 1

    return triangle, vertex, type
end

function action_space_size(env)
    return size(env.template, 2) * 2
end

function get_new_vertex_degree(edge_on_boundary)
    if edge_on_boundary
        return 4
    else
        return 6
    end
end

function PPO.step!(wrapper, action_index; no_action_reward=-4)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !env.is_terminated "Attempting to step in terminated environment with action $action_index"

    triangle, vertex, type = index_to_action(action_index)

    if TM.is_active_triangle(env.mesh, triangle)
        @assert type == 1 || type == 2

        if type == 1
            TM.step_flip!(env, triangle, vertex, no_action_reward)
        else
            edge_on_boundary = !TM.has_neighbor(env.mesh, triangle, vertex)
            new_vertex_degree = get_new_vertex_degree(edge_on_boundary)
            TM.step_split_allow_boundary!(env, triangle, vertex, no_action_reward, new_vertex_degree)
            # wrapper.desired_degree = env.d0
        end
    else
        TM.step_no_action!(env, no_action_reward)
    end
end


struct StateData
    state::Any
    num_half_edges::Any
end

function StateData()
    state = Matrix{Int}[]
    num_half_edges = Int[]
    StateData(state, num_half_edges)
end

function Base.length(s::StateData)
    return length(s.state)
end

function Base.show(io::IO, s::StateData)
    l = length(s)
    println(io, "StateData")
    println(io, "\t$l data points")
end

function PPO.update!(state_data::StateData, state)
    na = size(state, 2)
    push!(state_data.state, state)
    push!(state_data.num_half_edges, na)
end

function PPO.initialize_state_data(wrapper)
    return StateData()
end

function PPO.action_probabilities(policy, state)
    logits = vec(policy(state))
    p = softmax(logits)
    return p
end

function PPO.episode_state(state_data)
    return state_data
end

function PPO.episode_returns(rewards, state_data, discount)
    ne = length(rewards)

    values = zeros(ne)
    v = 0.0

    for idx = ne:-1:1
        v = rewards[idx] + discount * v
        values[idx] = v
    end

    return values
end

function pad_with_zeros(matrix, desired_ncols)
    nrows, ncols = size(matrix)
    @assert ncols <= desired_ncols

    return hcat(matrix, zeros(Int, nrows, desired_ncols - ncols))
end

function pad_and_concatenate(matrices, desired_ncols)
    padded_matrices = [pad_with_zeros(m, desired_ncols) for m in matrices]
    return cat(padded_matrices..., dims=3)
end

function negative_inf_mask(nrows, maxcols, maskcols)
    mask = zeros(Float32, nrows, maxcols)
    mask[:, maskcols+1:end] .= -Inf
    return mask
end

function PPO.batch_state(vec_state_data)
    num_half_edges = vcat([s.num_half_edges for s in vec_state_data]...)
    nhmax = maximum(num_half_edges)
    block = cat([pad_and_concatenate(s.state, nhmax) for s in vec_state_data]..., dims=3)
    mask = cat([negative_inf_mask(2, nhmax, nh) for nh in num_half_edges]..., dims=3)

    return block, mask
end

function PPO.batch_advantage(episodes)
    ret = PPO.rewards.(episodes)
    return vcat(ret...)
end

function PPO.batch_action_probabilities(policy, state)
    scores, mask = state
    logits = policy(scores) + mask
    nf, na, nb = size(logits)
    logits = reshape(logits, nf * na, nb)
    p = softmax(logits, dims=1)
    return p
end

function single_trajectory_return(wrapper, policy)
    env = wrapper.env

    done = PPO.is_terminal(wrapper)
    if done
        return 0.0
    else
        initial_score = env.current_score
        minscore = initial_score
        while !done
            probs = PPO.action_probabilities(policy, PPO.state(wrapper))
            action = rand(Categorical(probs))

            PPO.step!(wrapper, action)

            minscore = min(minscore, env.current_score)
            done = PPO.is_terminal(wrapper)
        end
        return initial_score - minscore
    end
end

function single_trajectory_normalized_return(wrapper, policy)
    env = wrapper.env
    maxreturn = env.current_score - env.opt_score
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_return(wrapper, policy)
        return ret / maxreturn
    end
end

function average_normalized_returns(wrapper, policy, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = single_trajectory_normalized_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_normalized_returns(wrapper, policy, num_attempts)
    maxret = -Inf
    for idx = 1:num_attempts
        initial_reset!(wrapper)
        ret = single_trajectory_normalized_return(wrapper, policy)
        maxret = max(maxret, ret)
    end
    return maxret
end

function average_best_normalized_returns(wrapper, policy, num_attempts, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_normalized_returns(wrapper, policy, num_attempts)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function single_trajectory_count_splits(wrapper, policy)
    done = PPO.is_terminal(wrapper)
    valid_splits = 0
    invalid_splits = 0
    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        triangle, vertex, type = index_to_action(action)
        if type == 2
            if TM.is_active_triangle(wrapper.env.mesh, triangle)
                valid_splits += 1
            else
                invalid_splits += 1
            end
        end

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)
    end
    return valid_splits, invalid_splits
end

function average_number_of_splits(wrapper, policy, num_trajectories)
    valid_splits = 0
    invalid_splits = 0
    for i = 1:num_trajectories
        PPO.reset!(wrapper)
        vs, is = single_trajectory_count_splits(wrapper, policy)
        valid_splits += vs
        invalid_splits += is
    end
    return valid_splits/num_trajectories, invalid_splits/num_trajectories
end

function actions_and_scores_history(wrapper, policy)
    actions = []
    scores = []
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        push!(actions, action)
        push!(scores, wrapper.env.current_score)

        done = PPO.is_terminal(wrapper)
    end
    return actions, scores
end

function best_policy_mesh!(wrapper, policy)
    initial_reset!(wrapper)
    actions, scores = actions_and_scores_history(wrapper, policy)
    idx = argmin(scores)
    initial_reset!(wrapper)
    for a in actions[1:idx]
        PPO.step!(wrapper, a)
    end
    s = (wrapper.env.initial_score - scores[idx])/(wrapper.env.initial_score - wrapper.env.opt_score)
    return s
end