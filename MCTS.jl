module MCTS

using BSON
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
    children::Vector{Node}
    visit_count::Vector{Int}
    total_action_values::Vector{Float64}
    mean_action_values::Vector{Float64}
    prior_probabilities::Vector{Float64}
    value::Float64
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

function has_child(n::Node, action)
    return isassigned(children(n), action)
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
    println(io, "Node")
    s = @sprintf "\tvalue = %1.3f" n.value
    println(io, s)
end

function Node(parent, action, reward, prior_probabilities, value)

    num_actions = length(prior_probabilities)

    children = Vector{Node}(undef, num_actions)
    visit_count = zeros(Int, num_actions)
    total_action_values = zeros(num_actions)
    mean_action_values = zeros(num_actions)

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
    )

    if !isnothing(parent)
        set_child!(parent, child, action)
    end

    return child
end

# constructor to make root node
function Node(prior_probabilities::Vector{Float64}, value::Float64)

    return Node(nothing, 0, 0.0, prior_probabilities, value)
end

function Node(env, policy)
    p, v = action_probabilities_and_value(policy, state(env))
    return Node(p, v)
end
##################################################################################################################################




##################################################################################################################################
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
##################################################################################################################################

function tree_policy_score(visit_count, action_values, prior_probabilities, settings)

    total_visits = sum(visit_count)
    score =
        exploration_factor(settings) *
        sqrt(total_visits) *
        (prior_probabilities ./ (1 .+ visit_count))
    score .+= action_values

    return score
end

function select_action_index(visit_count, action_values, prior_probabilities, settings)
    score = tree_policy_score(visit_count, action_values, prior_probabilities, settings)

    return rand(findall(score .== maximum(score)))
end

function select_action(n::Node, settings)
    idx = select_action_index(
        visit_count(n),
        mean_action_values(n),
        prior_probabilities(n),
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

function expand!(parent, action, env, policy)
    step!(env, action)
    probs, val = action_probabilities_and_value(policy, state(env))
    r = reward(env)
    child = Node(parent, action, r, probs, val)
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
        update = discount * value
        W[a] += update
        Q[a] = W[a] / N[a]

        reverse_step!(env, a)

        backup!(p, update, discount, env)
    else
        return
    end
end

function search!(root, env, policy, tree_settings)
    if !is_terminal(env)
        counter = 0

        while counter < maximum_iterations(tree_settings)
            node, action = select!(root, env, tree_settings)

            if !is_terminal(env)
                child = expand!(node, action, env, policy)
                backup!(child, value(child), discount(tree_settings), env)
            else
                v = reward(env)
                backup!(node, v, discount(tree_settings), env)
            end

            counter += 1
        end
    end
end

function mcts_action_probabilities(visit_count, temperature)
    ap = softmax(visit_count / temperature)
    return ap
end

function get_new_root(old_root, action, env, policy)
    if has_child(old_root, action)
        c = child(old_root, action)
        set_parent!(c, nothing)
        return c
    else
        p, v = action_probabilities_and_value(policy, state(env))
        new_root = Node(p, v)
        return new_root
    end
end

function tree_action_probabilities!(root, policy, env, tree_settings)
    search!(root, env, policy, tree_settings)
    ap = mcts_action_probabilities(visit_count(root), temperature(tree_settings))
    return ap
end

function step_mcts!(batch_data, root, env, policy, tree_settings)
    s = state(env)

    search!(root, env, policy, tree_settings)

    action_probs = mcts_action_probabilities(visit_count(root), temperature(tree_settings))
    action = rand(Categorical(action_probs))

    step!(env, action)
    r = reward(env)
    c = get_new_root(root, action, env, policy)
    t = is_terminal(env)

    update!(batch_data, s, action_probs, action, r, t)

    return c
end

function collect_sample_trajectory!(batch_data, env, policy, tree_settings)
    reset!(env)
    root = Node(env, policy)
    terminal = is_terminal(env)

    while !terminal
        root = step_mcts!(batch_data, root, env, policy, tree_settings)
        terminal = is_terminal(env)
    end
end

function collect_batch_data!(batch_data, env, policy, tree_settings, memory_size)
    while length(batch_data) < memory_size
        collect_sample_trajectory!(batch_data, env, policy, tree_settings)
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

##################################################################################################################################
struct PolicyIterationSettings
    discount
    l2_coeff
    memory_size
    batch_size
    num_epochs
    num_iter
    threshold
    num_samples
end

function number_of_samples(settings::PolicyIterationSettings)
    return settings.num_samples
end

function threshold(settings::PolicyIterationSettings)
    return settings.threshold
end

function l2_coefficient(settings::PolicyIterationSettings)
    return settings.l2_coeff
end

function memory_size(settings::PolicyIterationSettings)
    return settings.memory_size
end

function batch_size(settings::PolicyIterationSettings)
    return settings.batch_size
end

function number_of_epochs(settings::PolicyIterationSettings)
    return settings.num_epochs
end

function discount(settings::PolicyIterationSettings)
    return settings.discount
end

function number_of_iterations(settings::PolicyIterationSettings)
    return settings.num_iter
end
##################################################################################################################################

function step_batch!(policy, optimizer, state, target_probs, target_vals, l2_coeff)
    weights = params(policy)
    ce = mse = reg = l = 0.0
    grads = Flux.gradient(weights) do
        ce, mse, reg = loss(policy, state, target_probs, target_vals)
        reg = l2_coeff*reg
        l = ce + mse + reg
    end
    Flux.update!(optimizer, weights, grads)
    return ce, mse, reg
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

    cross_entropy = Float64[]
    value_mse = Float64[]
    weight_reg = Float64[]

    for start in range(1, step = batch_size, stop = memory_size)
        stop = min(start + batch_size - 1, memory_size)

        state = shuffled_state_data[start:stop]
        tp = target_probs[:, start:stop]
        tv = target_vals[start:stop]

        ce, mse, reg = step_batch!(policy, optimizer, state, tp, tv, l2_coeff)
        push!(cross_entropy, ce)
        push!(value_mse, mse)
        push!(weight_reg, reg)
    end

    return Flux.mean(cross_entropy), Flux.mean(value_mse), Flux.mean(weight_reg)
end

function train!(
    policy,
    optimizer,
    batch_data,
    settings;
    print_every = 10
)

    sd = state_data(batch_data)
    target_probs, target_vals = batch_target(batch_data, discount(settings))

    for epoch = 1:number_of_epochs(settings)
        cross_entropy, value_mse, weight_reg = step_epoch!(
            policy,
            optimizer,
            sd,
            target_probs,
            target_vals,
            batch_size(settings),
            l2_coefficient(settings),
        )
        if epoch % print_every == 0
            total = cross_entropy + value_mse + weight_reg
            @printf "\tCE : %1.4f\tMSE : %1.4f\tREG : %1.4f\tTOTAL : %1.4f\n" cross_entropy value_mse weight_reg total
        end
    end
end

function iterate!(policy, env, optimizer, evaluator, tree_settings, iter_settings; foldername = "")
    best_policy = deepcopy(policy)

    for iter in 1:number_of_iterations(iter_settings)
        data = BatchData(initialize_state_data(env))
        collect_batch_data!(data, env, best_policy, tree_settings, memory_size(iter_settings))

        train!(policy, optimizer, data, iter_settings)

        returns = evaluator(best_policy, policy, env, number_of_samples(iter_settings))

        mean_ret = Flux.mean(returns[2,:])
        @printf "\tMEAN RET = %1.4f\n" mean_ret

        new_win_rate = count(returns[2,:] .>= returns[1,:])/number_of_samples(iter_settings)

        if new_win_rate > threshold(iter_settings)
            filename = foldername * "policy-"*string(iter)*".bson"
            println("SAVING MODEL AT", filename)
            BSON.@save filename policy mean_ret
            best_policy = deepcopy(policy)
        end
    end

    return best_policy
end

# end module
end
# end module