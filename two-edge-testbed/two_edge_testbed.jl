function two_edge_game_env(; vertex_score = rand(-3:3, 5))
    t = [
        1 2 4
        1 3 2
        1 5 3
    ]
    p = rand(5, 2)
    mesh = EdgeFlip.Mesh(p, t)

    env = EdgeFlip.GameEnv(mesh, 0, d0 = mesh.d - vertex_score, maxflips = 2)
    return env
end

function step!(env::EdgeFlip.GameEnv, action)
    num_actions = EdgeFlip.number_of_actions(env)
    if 0 < action <= num_actions
        EdgeFlip.step!(env, action)
    else
        EdgeFlip.step!(env)
    end
end

function reset!(env::EdgeFlip.GameEnv; vertex_score = env.vertex_score)
    d0 = env.mesh.d - vertex_score
    EdgeFlip.reset!(env, d0 = d0)
end

function all_states(; template = -3:3, numrows = 5)
    nstates = length(template)
    rows = [repeat(repeat(template,inner=nstates^(k-1)),outer=nstates^(numrows-k)) for k = 1:numrows]
    return vcat(transpose.(rows)...)
end

function all_rewards(env,states)
    numstates = size(states,2)
    rewards = zeros(Int,9,numstates)
    for idx in 1:numstates
        state = states[:,idx]
        reset!(env,vertex_score = state)
        for a1 in 0:2
            r1 = a1 == 0 ? 0 : EdgeFlip.reward(env,a1)
            for a2 in 0:2
                reset!(env)
                step!(env, a1)
                r2 = a2 == 0 ? 0 : EdgeFlip.reward(env,a2)
                row = 3a1 + a2 + 1
                rewards[row,idx] = r1 + r2
            end
        end
    end
    return rewards
end