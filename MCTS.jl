module MCTS

using Distributions: Categorical

# methods that need to be overloaded
state(env) = nothing
step!(env, action) = nothing
reverse_step!(env, action) = nothing
reset!(env) = nothing
action_probabilities_and_value(policy, state) = nothing
reward(env) = nothing
update!(state_data, env) = nothing

mutable struct Node
    parent::Union{Nothing,Node}
    action::Int # integer action that brings me to this state from parent
    reward::Float64 # normalized reward for transitioning into this state
    children::Dict{Int,Node}
    visit_count::Dict{Int,Int}
    total_action_values::Dict{Int,Float64}
    mean_action_values::Dict{Int,Float64}
    prior_probabilities::Vector{Float64}
    value::Float64
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

function set_parent!(n::Node, p)
    n.parent = p
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

function value(n::Node)
    return n.value
end

function number_of_actions(n::Node)
    return length(prior_probabilities(n))
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

function Node(parent, action, reward, prior_probabilities, value, terminal)

    children = Dict{Int,Node}()
    visit_count = Dict{Int,Int}()
    total_action_values = Dict{Int,Float64}()
    mean_action_values = Dict{Int,Float64}()

    child = Node(
        parent,
        action,
        reward,
        children,
        visit_count,
        total_action_values,
        mean_action_values,
        prior_probabilities,
        value,
        terminal,
    )

    if !isnothing(parent)
        set_child!(parent, child, action)
    end

    return child
end

# constructor to make root node
function Node(prior_probabilities, value, terminal)

    return Node(
        nothing,
        0,
        0.0,
        prior_probabilities,
        value,
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

function select(node::Node, env, Cpuct)
    if is_terminal(node)
        return node, 0
    else
        action = select_action(node, Cpuct)
        if has_child(node, action)
            step!(env, action)
            c = child(node, action)
            return select(c, env, Cpuct)
        else
            return node, action
        end
    end
end

function expand!(parent, action, env, policy)
    step!(env, action)
    probs, val = action_probabilities_and_value(policy, state(env))
    terminal = is_terminal(env)
    r = reward(env)
    child = Node(parent, action, r, probs, val, terminal)
    return child
end

function parent_value(reward, value, discount)
    return reward + discount*value*(1 - reward)
end

function backup!(node, value, discount, env)
    if has_parent(node)
        p = parent(node)

        W = total_action_values(p)
        Q = mean_action_values(p)
        N = visit_count(p)
        a = action(node)
        r = reward(node)

        N[a] += 1
        update = parent_value(r, value, discount)
        W[a] += update
        Q[a] = W[a]/N[a]

        reverse_step!(env, a)
        
        backup!(p, update, discount, env)
    else
        return
    end
end

function move_to_root!(node, env)
    if has_parent(node)
        a = action(node)
        reverse_step!(env, a)
        move_to_root!(parent(node), env)
    else
        return node
    end
end

function search!(root, env, policy, Cpuct, discount, maxtime)
    if !is_terminal(root)
        start_time = time()
        elapsed = 0.0

        while elapsed < maxtime
            node, action = select(root, env, Cpuct)

            if !is_terminal(node)
                child = expand!(node, action, env, policy)
                backup!(child, value(child), discount, env)
            else
                backup!(node, value(node), discount, env)
            end

            elapsed = time() - start_time
        end
    end
end

function mcts_action_probabilities(visit_count, number_of_actions, temperature)
    action_probabilities = zeros(number_of_actions)
    for (key, value) in visit_count
        action_probabilities[key] = value
    end
    action_probabilities .^= (1.0/temperature)
    action_probabilities ./= sum(action_probabilities)
    return action_probabilities
end

function step_mcts!(batch_data, root, env, policy, Cpuct, discount, maxtime, temperature)
    s = state(env)
    
    search!(root, env, policy, Cpuct, discount, maxtime)
    na = number_of_actions(root)

    action_probs = mcts_action_probabilities(visit_count(root), na, temperature)
    action = rand(Categorical(action_probs))

    step!(env, action)
    r = reward(env)
    c = child(root, action)
    t = is_terminal(c)
    
    update!(batch_data, s, action_probs, action, r, t)

    set_parent!(c, nothing)
    
    return c
end

function collect_sample_trajectory!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)
    reset!(env)
    p, v = action_probabilities_and_value(policy, state(env))
    terminal = is_terminal(env)
    root = Node(p, v, terminal)

    while !terminal
        root = step_mcts!(batch_data, root, env, policy, Cpuct, discount, maxtime, temperature)
        terminal = is_terminal(root)
    end
end

function collect_batch_data!(batch_data, env, policy, Cpuct, discount, maxtime, temperature, batch_size)
    while length(batch_data) < batch_size
        collect_sample_trajectory!(batch_data, env, policy, Cpuct, discount, maxtime, temperature)
    end
end

function backup_state_value(rewards, terminal, discount)
    l = length(rewards)
    @assert l == length(terminal)
    
    values = zeros(l)
    counter = 0
    v = 0.0

    for idx = l:-1:1
        if terminal[idx]
            counter = 0
            v = 0.0
        end
        v = rewards[idx] + discount*v
        values[idx] = v
        counter += 1
    end

    return values
end

struct BatchData
    state_data::Any
    action_probabilities::Any
    actions
    rewards::Any
    terminal
    function BatchData(state_data)
        action_probabilities = Vector{Float64}[]
        actions = Int64[]
        rewards = Float64[]
        terminal = Bool[]
        new(state_data, action_probabilities, actions, rewards, terminal)
    end
end

function Base.length(data::BatchData)
    return length(data.action_probabilities)
end

function Base.show(io::IO, data::BatchData)
    num_data = length(data)
    println(io, "BatchData")
    println(io, "\t$num_data data points")
end

function update!(batch_data::BatchData, state, action_probabilities, action, reward, terminal)
    update!(batch_data.state_data, state)
    push!(batch_data.action_probabilities, action_probabilities)
    append!(batch_data.actions, action)
    append!(batch_data.rewards, reward)
    append!(batch_data.terminal, terminal)
end

# end module
end
# end module