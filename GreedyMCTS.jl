module GreedyMCTS

using Flux: softmax
using Distributions: Categorical
using Printf

# methods that need to be overloaded
# state(env) = nothing
step!(env, action) = nothing
reverse_step!(env, action) = nothing
is_terminal(env) = nothing
reward(env) = nothing
estimate_value(env) = nothing

mutable struct Node
    parent::Union{Nothing,Node}
    action::Int # integer action that brings me to this state from parent
    num_actions::Int
    children::Vector{Node}
    visit_count::Vector{Int}
    total_action_values::Vector{Float64}
    mean_action_values::Vector{Float64}
    value::Float64
end

function set_parent!(n::Node, p)
    n.parent = p
end

function children(n::Node)
    return n.children
end

function child(n::Node, action)
    return n.children[action]
end

function has_child(n::Node, action)
    return isassigned(children(n), action)
end

function set_child!(parent, child, action)
    @assert !has_child(parent, action)
    parent.children[action] = child
    parent.visit_count[action] = 0
    parent.total_action_values[action] = 0.0
    parent.mean_action_values[action] = 0.0
end

function num_children(n::Node)
    return length(children(n))
end

function number_of_actions(n::Node)
    return n.num_actions
end

function parent(n::Node)
    return n.parent
end

function has_parent(n::Node)
    return !isnothing(parent(n))
end

function value(n::Node)
    return n.value
end

function mean_action_values(n::Node)
    return n.mean_action_values
end

function total_action_values(n::Node)
    return n.total_action_values
end

function visit_count(n::Node)
    return n.visit_count
end

function action(n::Node)
    return n.action
end

function Base.show(io::IO, n::Node)
    println(io, "Node")
    s = @sprintf "\tvalue = %1.3f" n.value
    println(io, s)
end

function Node(parent, action, num_actions, value)

    children = Vector{Node}(undef, num_actions)
    visit_count = zeros(Int, num_actions)
    total_action_values = zeros(num_actions)
    mean_action_values = zeros(num_actions)

    child = Node(
        parent,
        action,
        num_actions,
        children,
        visit_count,
        total_action_values,
        mean_action_values,
        value,
    )

    if !isnothing(parent)
        set_child!(parent, child, action)
    end

    return child
end

# Constructor to make root Node
function Node(num_actions::Int, value::Float64)
    return Node(nothing, 0, num_actions, value)
end

function Node(env)
    value = estimate_value(env)
    num_actions = number_of_actions(env)
    return Node(num_actions, value)
end

struct TreeSettings
    exploration_factor::Any
    maximum_iterations::Any
    temperature::Any
    discount::Any
end

function exploration_factor(settings::TreeSettings)
    return settings.exploration_factor
end

function maximum_iterations(settings::TreeSettings)
    return settings.maximum_iterations
end

function temperature(settings::TreeSettings)
    return settings.temperature
end

function discount(settings::TreeSettings)
    return settings.discount
end

function tree_policy_score(action_values, visit_count, settings)
    log_total_visits = log(sum(visit_count))
    score = copy(action_values)

    for (action, visit) in enumerate(visit_count)
        if visit == 0
            score[action] = Inf
        else
            score[action] += exploration_factor(settings)*sqrt(log_total_visits/visit)
        end
    end

    return score
end


function select_action_index(action_values, visit_count, settings)
    score = tree_policy_score(action_values, visit_count, settings)
    return rand(findall(score .== maximum(score)))
end

function select_action(n::Node, settings)
    idx = select_action_index(
        mean_action_values(n),
        visit_count(n),
        settings,
    )
    return idx
end

function select!(node::Node, env, settings)
    if is_terminal(env)
        return node, 0
    else
        action = select_action(node, settings)
        if has_child(node, action)
            step!(env, action)
            c = child(node, action)
            return select!(c, env, settings)
        else
            return node, action
        end
    end
end

function expand!(parent, action, env)
    step!(env, action)
    na = number_of_actions(env)
    value = estimate_value(env)
    child = Node(parent, action, na, value)
    return child
end

function backup!(node, value, discount, env)
    if has_parent(node)
        p = parent(node)

        W = total_action_values(p)
        Q = mean_action_values(p)
        N = visit_count(p)
        a = action(node)
        
        N[a] += 1
        update = discount*value
        W[a] += update
        Q[a] = W[a]/N[a]

        reverse_step!(env, a)

        backup!(p, update, discount, env)
    end
end

function search!(root, env, tree_settings)
    if !is_terminal(env)
        for counter = 1:maximum_iterations(tree_settings)
            node, action = select!(root, env, tree_settings)

            if !is_terminal(env)
                child = expand!(node, action, env)
                backup!(child, value(child), discount(tree_settings), env)
            else
                backup!(node, value(node), discount(tree_settings), env)
            end
        end
    end
end

function mcts_action_probabilities(visit_count, temperature)
    vs = Float64.(visit_count)
    vs[vs .== 0.0] .= -Inf
    return softmax(vs/temperature)
end

function get_new_root(old_root, action, env)
    @assert has_child(old_root, action)
    
    c = child(old_root, action)
    set_parent!(c, nothing)
    return c
end

function tree_action_probabilities!(root, env, tree_settings)
    search!(root, env, tree_settings)
    ap = mcts_action_probabilities(visit_count(root), temperature(tree_settings))
    return ap
end

function step_mcts!(root, env, tree_settings)
    ap = tree_action_probabilities!(root, env, tree_settings)
    action = rand(Categorical(ap))

    step!(env, action)
    child = get_new_root(root, action, env)

    return child
end

end