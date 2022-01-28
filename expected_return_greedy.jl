function env_score(vs)
    return sum(abs.(vs),dims=1)
end

function greedy_reward(states)
    d = flip_reward(states)
    g = [max(0,i) for i in d]
    return g
end


g = greedy_reward(states)
avg_reward = sum(g)/length(g)
