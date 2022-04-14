module MCTS

using OrderedCollections

struct Node
    parent::Union{Nothing,Node}
    action::Int # integer action that brings me to this state from parent
    children::OrderedDict{Int,Node}
    visit_count::OrderedDict{Int,Int}
    total_action_values::OrderedDict{Int,Float64}
    mean_action_values::OrderedDict{Int,Float64}
    prior_probabilities::Vector{Float64}
    terminal::Bool
end

function num_children(n::Node)
    return length(children(n))
end

function num_actions(n::Node)
    return length(prior_probabilities(n))
end

function parent(n::Node)
    return n.parent
end

function children(n::Node)
    return n.children
end

function child(n::Node, action)
    return n.children[action]
end

function has_children(n::Node)
    return !isempty(children(n))
end

function has_child(n::Node, action)
    return haskey(children(n), action)
end

function prior_probabilities(n::Node)
    return n.prior_probabilities
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

function Base.show(io::IO, n::Node)
    nc = num_children(n)
    println(io, "Node")
    println(io, "\t$nc children")
end

# constructor to make root node
function Node(prior_probabilities; terminal = false)

    children = OrderedDict{Int,Node}()
    visit_count = OrderedDict{Int,Int}()
    total_action_values = OrderedDict{Int,Float64}()
    mean_action_values = OrderedDict{Int,Float64}()

    Node(
        nothing,
        0,
        children,
        visit_count,
        total_action_values,
        mean_action_values,
        prior_probabilities,
        terminal,
    )
end

function PUCT_exploration(prior_probs, visit_count, Cpuct)
    u = Cpuct * prior_probs * sqrt(sum(visit_count)) ./ (1 .+ visit_count)
    return u
end

function PUCT_exploration(prior_probs, visit_count, Cpuct)

    u = Cpuct * prior_probs * sqrt(sum(values(visit_count)))
    for (action, visit) in visit_count
        u[action] /= (1 + visit)
    end
    return u
end

function PUCT_score(prior_probs, visit_count, action_values, Cpuct)
    score = PUCT_exploration(prior_probs, visit_count, Cpuct)
    for (action, value) in action_values
        score[action] += value
    end
    return score
end

function select_action_index(prior_probs, visit_count, action_values, Cpuct)
    s = PUCT_score(prior_probs, visit_count, action_values, Cpuct)
    idx = rand(findall(s .== maximum(s)))
    return idx
end

function select_action(n::Node, Cpuct)
    idx = select_action_index(
        prior_probabilities(n),
        visit_count(n),
        mean_action_values(n),
        Cpuct,
    )
    return idx
end

function select(n::Node, env, Cpuct)
    action = select_action(n, Cpuct)
    if has_child(n, action)
        step!(env, action)
        c = child(n, action)
        select(c, env, Cpuct)
    else
        return n, action
    end
end


# end module
end
# end module