module GreedyMCTS

using Flux: softmax
using Distributions: Categorical
using Printf

# methods that need to be overloaded
state(env) = nothing
step!(env, action) = nothing
reverse_step!(env, action) = nothing
reset!(env) = nothing
is_terminal(env) = nothing
reward(env) = nothing

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
function Node(num_actions, value)
    return Node(nothing, 0, num_actions, value)
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

function probability_weight(settings::TreeSettings)
    return settings.probability_weight
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
            score[visit] = Inf
        else
            score[visit] += sqrt(log_total_visits/visit)
        end
    end

    return score
end