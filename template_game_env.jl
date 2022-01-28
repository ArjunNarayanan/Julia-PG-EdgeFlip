function env_score(vs)
    return sum(abs.(vs))
end

mutable struct TemplateEnv
    vertex_score::Any
    score::Any
    reward::Any
    is_terminated::Any
    function TemplateEnv(vs)
        @assert length(vs) == 4
        score = env_score(vs)
        new(vs, score, 0.0, false)
    end
end

function TemplateEnv()
    vs = rand([-1, 0, 1], 4)
    return TemplateEnv(vs)
end

function reset!(env::TemplateEnv)
    env.vertex_score = rand([-1, 0, 1], 4)
    env.score = env_score(env.vertex_score)
    env.reward = 0.0
    env.is_terminated = false
end

function step!(env::TemplateEnv, action; no_flip_reward = 0.0)
    if action == 1
        env.reward = no_flip_reward
        env.is_terminated = true
    elseif action == 2
        old_score = env.score
        vs = env.vertex_score + [-1, -1, 1, 1]
        new_score = env_score(vs)

        env.vertex_score = vs
        env.score = new_score
        env.reward = old_score - new_score
        env.is_terminated = true
    end
end

function state(env::TemplateEnv)
    return env.vertex_score
end

function reward(env::TemplateEnv)
    return env.reward
end

function is_terminated(env::TemplateEnv)
    return env.is_terminated
end

function all_states()
    template = [-1,0,1]

    row1 = repeat(template,27)
    row2 = repeat(repeat(template,inner=3),outer=9)
    row3 = repeat(repeat(template,inner=9),outer=3)
    row4 = repeat(template,inner=27)

    states = vcat(row1',row2',row3',row4')
end

function flip_reward(states)
    old_score = mapslices(env_score,states,dims=1)
    delta = [-1,-1,1,1]
    newstates = states .+ delta
    new_score = mapslices(env_score,newstates,dims=1)

    return old_score - new_score
end
