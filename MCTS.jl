module MCTS

using Flux
using Distributions: Categorical
using Printf
using Random

# methods that need to be overloaded
state(env) = nothing
step!(env, action) = nothing
reverse_step!(env, action) = nothing
reset!(env) = nothing
is_terminal(env) = nothing
reward(env) = nothing

action_probabilities_and_value(policy, state) = nothing
batch_action_logprobs_and_values(policy, state) = nothing
update!(state_data, env) = nothing
reorder(state_data, idx) = nothing
initialize_state_data(env) = nothing
batch_state(state_data) = nothing

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

# function is_terminal(n::Node)
#     return n.terminal
# end

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

    num_actions = length(prior_probabilities)

    children = Dict{Int,Node}()
    visit_count = Dict{Int,Int}(zip(1:num_actions, ones(Int, num_actions)))
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

    return Node(nothing, 0, 0.0, prior_probabilities, value, terminal)
end

# function PUCT_exploration(prior_probs, visit_count, Cpuct)

#     u = Cpuct * prior_probs * sqrt(sum(values(visit_count)))
#     for (action, visit) in visit_count
#         u[action] /= (1 + visit)
#     end
#     return u
# end

# function PUCT_score(prior_probs, visit_count, action_values, Cpuct)
#     score = PUCT_exploration(prior_probs, visit_count, Cpuct)
#     for (action, value) in action_values
#         score[action] += value
#     end
#     return score
# end

# function select_action_index(prior_probs, visit_count, action_values, Cpuct)
#     s = PUCT_score(prior_probs, visit_count, action_values, Cpuct)
#     idx = rand(findall(s .== maximum(s)))
#     return idx
# end

# function select_action(n::Node, Cpuct)
#     idx = select_action_index(
#         prior_probabilities(n),
#         visit_count(n),
#         mean_action_values(n),
#         Cpuct,
#     )
#     return idx
# end

# function select!(node::Node, env, Cpuct)
#     if is_terminal(env)
#         return node, 0
#     else
#         action = select_action(node, Cpuct)
#         if has_child(node, action)
#             step!(env, action)
#             c = child(node, action)
#             return select!(c, env, Cpuct)
#         else
#             return node, action
#         end
#     end
# end

function tree_policy_score(
    visit_count,
    action_values,
    prior_probabilities,
    tree_exploration_factor,
    probability_weight,
)

    score = probability_weight * prior_probabilities
    log_total_visit_count = log(sum(values(visit_count)))

    for (action, visit) in visit_count
        score[action] /= (1 + visit)
        score[action] += tree_exploration_factor * sqrt(log_total_visit_count / visit)
    end

    for (action, value) in action_values
        score[action] += value
    end

    return score
end

function select_action_index(
    visit_count,
    action_values,
    prior_probabilities,
    tree_exploration_factor,
    probability_weight,
)
    score = tree_policy_score(
        visit_count,
        action_values,
        prior_probabilities,
        tree_exploration_factor,
        probability_weight,
    )
    return rand(findall(score .== maximum(score)))
end

function select_action(n::Node, tree_exploration_factor, probability_weight)
    idx = select_action_index(
        visit_count(n),
        mean_action_values(n),
        prior_probabilities(n),
        tree_exploration_factor,
        probability_weight,
    )
    return idx
end

function select!(node::Node, env, tree_exploration_factor, probability_weight)
    if is_terminal(env)
        return node, 0
    else
        action = select_action(node, tree_exploration_factor, probability_weight)
        if has_child(node, action)
            step!(env, action)
            c = child(node, action)
            return select!(c, env, tree_exploration_factor, probability_weight)
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
    return discount * value
    # return reward + discount * value * (1 - reward)
end

function backup!(node, value, discount, env)
    if has_parent(node)
        p = parent(node)

        W = total_action_values(p)
        Q = mean_action_values(p)
        N = visit_count(p)
        a = action(node)
        r = reward(node)

        update = parent_value(r, value, discount)
        W[a] += update
        Q[a] = W[a] / N[a]
        N[a] += 1

        reverse_step!(env, a)

        backup!(p, update, discount, env)
    else
        return
    end
end

function search!(
    root,
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
)
    if !is_terminal(env)
        counter = 0

        while counter < maxiter
            node, action = select!(root, env, tree_exploration_factor, probability_weight)

            if !is_terminal(env)
                child = expand!(node, action, env, policy)
                backup!(child, value(child), discount, env)
            else
                v = reward(env)
                backup!(node, v, discount, env)
            end

            counter += 1
        end
    end
end

function mcts_action_probabilities(visit_count, number_of_actions, temperature)
    vs = zeros(number_of_actions)
    for (key, value) in visit_count
        vs[key] = value
    end
    ap = softmax(vs / temperature)
    return ap
end

function get_new_root(old_root, action, env, policy)
    if has_child(old_root, action)
        c = child(old_root, action)
        set_parent!(c, nothing)
        return c
    else
        p, v = action_probabilities_and_value(policy, state(env))
        new_root = Node(p, v, is_terminal(env))
        return new_root
    end
end

# function tree_action_probabilities(
#     policy,
#     env,
#     tree_exploration_factor,
#     probability_weight,
#     discount,
#     maxtime,
#     temperature,
# )
#     p, v = action_probabilities_and_value(policy, state(env))
#     root = Node(p, v, is_terminal(env))
#     search!(
#         root,
#         env,
#         policy,
#         tree_exploration_factor,
#         probability_weight,
#         discount,
#         maxtime,
#     )
#     na = number_of_actions(root)

#     action_probs = mcts_action_probabilities(visit_count(root), na, temperature)
#     return action_probs
# end

function tree_action_probabilities!(
    root,
    policy,
    env,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
)
    search!(
        root,
        env,
        policy,
        tree_exploration_factor,
        probability_weight,
        discount,
        maxiter,
    )
    na = number_of_actions(root)

    ap = mcts_action_probabilities(visit_count(root), na, temperature)
    return ap
end

function step_mcts!(
    batch_data,
    root,
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
)
    s = state(env)

    search!(
        root,
        env,
        policy,
        tree_exploration_factor,
        probability_weight,
        discount,
        maxiter,
    )
    na = number_of_actions(root)

    action_probs = mcts_action_probabilities(visit_count(root), na, temperature)
    action = rand(Categorical(action_probs))

    step!(env, action)
    r = reward(env)
    c = get_new_root(root, action, env, policy)
    t = is_terminal(env)

    update!(batch_data, s, action_probs, action, r, t)

    return c
end

function collect_sample_trajectory!(
    batch_data,
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
)
    reset!(env)
    p, v = action_probabilities_and_value(policy, state(env))
    terminal = is_terminal(env)
    root = Node(p, v, terminal)

    while !terminal
        root = step_mcts!(
            batch_data,
            root,
            env,
            policy,
            tree_exploration_factor,
            probability_weight,
            discount,
            maxiter,
            temperature,
        )
        terminal = is_terminal(env)
    end
end

function collect_batch_data!(
    batch_data,
    env,
    policy,
    tree_exploration_factor,
    probability_weight,
    discount,
    maxiter,
    temperature,
    batch_size,
)
    while length(batch_data) < batch_size
        collect_sample_trajectory!(
            batch_data,
            env,
            policy,
            tree_exploration_factor,
            probability_weight,
            discount,
            maxiter,
            temperature,
        )
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
        v = rewards[idx] + discount * v
        values[idx] = v
        counter += 1
    end

    return values
end

struct BatchData
    state_data::Any
    action_probabilities::Any
    actions::Any
    rewards::Any
    terminal::Any
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

function state_data(data::BatchData)
    return data.state_data
end

function Base.show(io::IO, data::BatchData)
    num_data = length(data)
    println(io, "BatchData")
    println(io, "\t$num_data data points")
end

function update!(
    batch_data::BatchData,
    state,
    action_probabilities,
    action,
    reward,
    terminal,
)
    update!(batch_data.state_data, state)
    push!(batch_data.action_probabilities, action_probabilities)
    append!(batch_data.actions, action)
    append!(batch_data.rewards, reward)
    append!(batch_data.terminal, terminal)
end

function batch_target(b::BatchData, discount)
    action_probabilities = cat(b.action_probabilities..., dims = 2)
    state_values = backup_state_value(b.rewards, b.terminal, discount)

    return action_probabilities, state_values
end

function loss_components(policy_logprobs, policy_vals, target_probs, target_vals, weights)

    @assert size(policy_logprobs) == size(target_probs)
    @assert length(policy_vals) == length(target_vals)

    p1 = Flux.mean(-vec(sum(target_probs .* (policy_logprobs), dims = 1)))
    p2 = Flux.mean(abs2, (policy_vals - target_vals))
    sqnorm(w) = sum(abs2, w)
    p3 = sum(sqnorm, weights)

    return p1, p2, p3
end

function loss(policy, state, target_probabilities, target_values)
    policy_logprobs, policy_vals = batch_action_logprobs_and_values(policy, state)

    @assert size(policy_logprobs) == size(target_probabilities)
    @assert length(policy_vals) == length(target_values)
    weights = params(policy)

    p1, p2, p3 = loss_components(
        policy_logprobs,
        policy_vals,
        target_probabilities,
        target_values,
        weights,
    )

    return p1, p2, p3
end

function step_batch!(policy, optimizer, state, target_probs, target_vals, l2_coeff)
    weights = params(policy)
    ce = mse = reg = l = 0.0
    grads = Flux.gradient(weights) do
        ce, mse, reg = loss(policy, state, target_probs, target_vals)
        l = ce + mse + l2_coeff * reg
    end
    # @printf "\tCE : %1.4f\tMSE : %1.4f\tREG : %1.4f\tTOTAL : %1.4f\n" ce mse l2_coeff * reg l
    Flux.update!(optimizer, weights, grads)
end

function step_epoch!(
    policy,
    optimizer,
    state_data,
    target_probs,
    target_vals,
    batch_size,
    l2_coeff,
)
    memory_size = length(state_data)
    @assert size(target_probs, 2) == memory_size
    @assert length(target_vals) == memory_size
    @assert memory_size >= batch_size

    idx = randperm(memory_size)
    shuffled_state_data = reorder(state_data, idx)
    target_probs = target_probs[:, idx]
    target_vals = target_vals[idx]

    for start in range(1, step = batch_size, stop = memory_size)
        stop = min(start + batch_size - 1, memory_size)

        state = shuffled_state_data[start:stop]
        tp = target_probs[:, start:stop]
        tv = target_vals[start:stop]

        step_batch!(policy, optimizer, state, tp, tv, l2_coeff)
    end
end

function train!(
    policy,
    env,
    optimizer,
    batch_data,
    discount,
    batch_size,
    l2_coeff,
    num_epochs,
    evaluator;
    evaluate_every = 50,
)

    ret = evaluator(policy, env)

    sd = state_data(batch_data)
    target_probs, target_vals = batch_target(batch_data, discount)

    for epoch = 1:num_epochs
        step_epoch!(policy, optimizer, sd, target_probs, target_vals, batch_size, l2_coeff)

        if epoch % evaluate_every == 0
            new_rets = evaluator(policy, env)
        end
    end
end

# end module
end
# end module