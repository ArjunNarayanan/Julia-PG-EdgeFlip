module MCTS

using OrderedCollections

state(env) = nothing
step!(env, action) = nothing
reverse_step!(env, action) = nothing
reset!(env) = nothing
action_probabilities_and_value(policy, state) = nothing
is_terminal(env) = nothing
reward(env) = nothing


struct Node
    parent::Union{Nothing,Node}
    action::Int # integer action that brings me to this state from parent
    reward::Float64 # normalized reward for transitioning into this state
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

function set_child!(parent, child, action)
    @assert !has_child(parent, action)
    parent.children[action] = child
    parent.visit_count[action] = 0
    parent.total_action_values[action] = 0.0
    parent.mean_action_values[action] = 0.0
end

function has_children(n::Node)
    return !isempty(children(n))
end

function has_child(n::Node, action)
    return haskey(children(n), action)
end

function has_parent(n::Node)
    return !isnothing(parent(n))
end

function prior_probabilities(n::Node)
    return n.prior_probabilities
end

function is_terminal(n::Node)
    return n.terminal
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

function reward(n::Node)
    return n.reward
end

function Base.show(io::IO, n::Node)
    nc = num_children(n)
    println(io, "Node")
    println(io, "\t$nc children")
end

function Node(parent, action, reward, prior_probabilities, terminal)

    children = OrderedDict{Int,Node}()
    visit_count = OrderedDict{Int,Int}()
    total_action_values = OrderedDict{Int,Float64}()
    mean_action_values = OrderedDict{Int,Float64}()

    child = Node(
        parent,
        action,
        reward,
        children,
        visit_count,
        total_action_values,
        mean_action_values,
        prior_probabilities,
        terminal,
    )

    if !isnothing(parent)
        set_child!(parent, child, action)
    end

    return child
end

# constructor to make root node
function Node(prior_probabilities; terminal = false)

    return Node(
        nothing,
        0,
        0.0,
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

function expand(parent, action, env, policy)
    step!(env, action)
    probs, val = action_probabilities_and_value(policy, state(env))
    terminal = is_terminal(env)
    r = reward(env)
    child = Node(parent, action, r, probs, terminal)
    return child, val
end

function backup(node, value, discount, env)
    if has_parent(node)
        p = parent(node)

        W = total_action_values(p)
        Q = mean_action_values(p)
        N = visit_count(p)
        a = action(node)
        r = reward(node)

        N[a] += 1
        update = r + discount*value*(1-r)
        W[a] += update
        Q[a] = W[a]/N[a]

        reverse_step!(env, a)
        
        backup(p, update, discount, env)
    else
        return node
    end
end

# end module
end
# end module