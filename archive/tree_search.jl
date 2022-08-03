module TreeSearch

using Random
using EdgeFlip

struct Node
    parent::Union{Nothing,Node}
    children::Vector{Node}
    data::Any
end

function num_children(n::Node)
    return length(children(n))
end

function children(n::Node)
    return n.children
end

function children(n::Node, idx)
    return n.children[idx]
end

function parent(n::Node)
    return n.parent
end

function has_children(n::Node)
    return num_children(n) > 0
end

function has_parent(n::Node)
    return !isnothing(parent(n))
end

function Base.show(io::IO, n::Node)
    nc = num_children(n)
    println(io, "Node")
    println(io, "   $nc children")
end

function initialize_data()
    data = Dict(
        :actions => Int[],
        :rewards => Float64[],
        :terminal => false,
        :reverse_action => 0,
        :level => 0,
        :best_child => 0,
        :max_return => 0.0,
    )
    return data
end

function actions(n::Node)
    return n.data[:actions]
end

function actions(n::Node, idx)
    return n.data[:actions][idx]
end

function rewards(n::Node)
    return n.data[:rewards]
end

function rewards(n::Node, idx)
    return n.data[:rewards][idx]
end

function is_terminal(n::Node)
    return n.data[:terminal]
end

function initialize_children()
    children = Vector{Node}(undef, 0)
    return children
end

function best_child(n::Node)
    return n.data[:best_child]
end

function max_return(n::Node)
    return n.data[:max_return]
end

# create a root node
function Node(; terminal = false)
    children = initialize_children()
    data = initialize_data()
    data[:terminal] = terminal
    Node(nothing, children, data)
end

# create a node with the given parent.
# update parent accordingly
function child!(parent::Node)
    children = initialize_children()
    data = initialize_data()
    child = Node(parent, children, data)
    push!(parent.children, child)
    return child
end

function add_child!(parent::Node, action, reward, terminal)
    child = child!(parent)

    push!(parent.data[:actions], action)
    push!(parent.data[:rewards], reward)

    child.data[:reverse_action] = action
    child.data[:terminal] = terminal
    child.data[:level] = parent.data[:level] + 1

    return child
end

function all_rewards(env)
    return [EdgeFlip.reward(env, a) for a = 1:EdgeFlip.number_of_actions(env)]
end

function is_terminal(env, action)
    EdgeFlip.step!(env, action)
    terminal = EdgeFlip.is_terminated(env)
    EdgeFlip.reverse_step!(env, action)
    return terminal
end

function expand!(parent, env, max_branching_factor)
    rewards = all_rewards(env)
    actions = findall(rewards .== maximum(rewards))

    if length(actions) > max_branching_factor
        idx = randperm(length(actions))[1:max_branching_factor]
        actions = actions[idx]
    end

    rewards = rewards[actions]
    terminated = [is_terminal(env, a) for a in actions]
    for (a, r, t) in zip(actions, rewards, terminated)
        add_child!(parent, a, r, t)
    end
end

function level_down!(parent, env, child_idx)
    @assert num_children(parent) >= child_idx > 0
    action = parent.data[:actions][child_idx]
    EdgeFlip.step!(env, action)
    child = children(parent, child_idx)
    return child
end

function level_up!(child, env)
    p = parent(child)
    action = child.data[:reverse_action]
    EdgeFlip.reverse_step!(env, action)
    return p
end

function grow_at!(node, env, tree_depth, max_branching_factor)
    if has_children(node)
        for idx = 1:num_children(node)
            child = level_down!(node, env, idx)
            grow_at!(child, env, tree_depth, max_branching_factor)
            level_up!(child, env)
        end
    elseif node.data[:level] < tree_depth && !node.data[:terminal]
        expand!(node, env, max_branching_factor)
        grow_at!(node, env, tree_depth, max_branching_factor)
    else
        return
    end
end

function collect_returns!(node)
    if has_children(node)
        ret = rewards(node) + collect_returns!.(children(node))
        maxidx = rand(findall(ret .== maximum(ret)))
        maxr = ret[maxidx]
        node.data[:best_child] = maxidx
        node.data[:max_return] = maxr
        return maxr
    else
        return 0.0
    end
end

function step_best_trajectory!(env, node)
    terminal = is_terminal(node)
    while !terminal && has_children(node)
        idx = best_child(node)
        action = actions(node, idx)
        EdgeFlip.step!(env, action)
        node = children(node, idx)
        terminal = is_terminal(node)
    end
end

function step_tree_search!(env, tree_depth, max_branching_factor)
    root = Node()
    grow_at!(root, env, tree_depth, max_branching_factor)
    collect_returns!(root)
    maxr = max_return(root)
    step_best_trajectory!(env, root)
    return maxr
end

function single_trajectory_return(env, tree_depth, max_branching_factor)
    done = EdgeFlip.is_terminated(env)
    if done
        return 0.0
    else
        initial_score = EdgeFlip.score(env)
        minscore = initial_score
        while !done
            step_tree_search!(env, tree_depth, max_branching_factor)
            minscore = min(minscore, EdgeFlip.score(env))
            done = EdgeFlip.is_terminated(env)
        end
        return initial_score - minscore
    end
end

function single_trajectory_normalized_return(env, tree_depth, max_branching_factor)
    maxreturn = EdgeFlip.score(env) - env.optimum_score
    if maxreturn == 0
        return 1.0
    else
        ret = single_trajectory_return(env, tree_depth, max_branching_factor)
        return ret / maxreturn
    end
end

function normalized_returns(env, tree_depth, max_branching_factor, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        EdgeFlip.reset!(env)
        ret[idx] = single_trajectory_normalized_return(env, tree_depth, max_branching_factor)
    end
    return ret
end

function average_normalized_returns(env, tree_depth, max_branching_factor, num_trajectories)
    ret = normalized_returns(env, tree_depth, max_branching_factor, num_trajectories)
    return sum(ret) / length(ret)
end

end