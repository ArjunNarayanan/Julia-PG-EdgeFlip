using EdgeFlip

function all_states()
    template = -3:1:3

    row1 = repeat(template, 2401)
    row2 = repeat(repeat(template, inner = 7), 343)
    row3 = repeat(repeat(template, inner = 49), 49)
    row4 = repeat(repeat(template, inner = 343), 7)
    row5 = repeat(template, inner = 2401)
    return vcat(row1', row2', row3', row4', row5')
end

function get_state_rewards(mesh0, state)
    mesh = deepcopy(mesh0)
    env = EdgeFlip.GameEnv(mesh, 0, d0 = mesh.d - state, fixed_reset = true)

    EdgeFlip.step!(env, 5)
    r1 = EdgeFlip.reward(env)
    EdgeFlip.step!(env, 4)
    r2 = max(0, EdgeFlip.reward(env))

    EdgeFlip.reset!(env)

    EdgeFlip.step!(env, 4)
    p1 = EdgeFlip.reward(env)
    EdgeFlip.step!(env, 5)
    p2 = max(0, EdgeFlip.reward(env))

    return r1, r2, p1, p2
end

function all_state_rewards(mesh, states)
    num_states = size(states, 2)

    R = zeros(Int, 2, num_states)
    P = zeros(Int, 2, num_states)

    for i = 1:num_states
        r1, r2, p1, p2 = get_state_rewards(mesh, states[:, i])
        R[:, i] .= [r1, r2]
        P[:, i] .= [p1, p2]
    end
    return R, P
end

t = [
    1 2 5
    2 4 5
    2 3 4
]
p = rand(5, 2)
mesh = EdgeFlip.Mesh(p, t)

states = all_states()
num_states = size(states, 2)
R, P = all_state_rewards(mesh, states)

sumR = vec(sum(R,dims=1))
sumP = vec(sum(P,dims=1))

non_greedy_indices = findall((sumR .> 0) .& (sumR .> sumP) .& (R[1,:] .< P[1,:]))
states_with_gain = findall((sumR .> 0) .| (sumP .> 0))

percent_non_greedy = length(non_greedy_indices)/length(states_with_gain)


non_greedy_states = [-3  -2  -3  -2  -3  -2  -3  -2  -3  -2  -3  -2
                      2   2   3   3   2   2   3   3   2   2   3   3
                     -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1  -1
                      0   0   0   0   0   0   0   0   0   0   0   0
                     -3  -3  -3  -3  -2  -2  -2  -2  -1  -1  -1  -1]