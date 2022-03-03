using PyPlot
using EdgeFlip

function average_maxscore(env, ntrials)
    maxscore = 0
    for trial = 1:ntrials
        EdgeFlip.reset!(env)
        maxscore += EdgeFlip.score(env)
    end
    return maxscore / ntrials
end

function maxscore_vs_nflips(env, nflip_range, ntrials; maxstepfactor = 1.2)
    maxscore = zeros(length(nflip_range))
    for (idx, nflip) in enumerate(nflip_range)
        maxflips = ceil(Int, maxstepfactor * nflip)
        EdgeFlip.reset!(env, nflips = nflip, maxflips = maxflips)
        maxscore[idx] = average_maxscore(env, ntrials)
    end
    return maxscore
end

function plot_maxscore(nflips, maxscore; filename = "")
    fig, ax = subplots()
    ax.plot(nflips, maxscore)
    ax.set_xlabel("normalized number of initial flips")
    ax.set_ylabel("average maximum score")
    if length(filename) > 0
        fig.savefig(filename)
    end
    return fig
end

nref = 1
nflips = 1:42

env = EdgeFlip.GameEnv(nref, 0)
maxscore = maxscore_vs_nflips(env, nflips, 100)

normalized_nflips = nflips ./ EdgeFlip.number_of_actions(env)
plot_maxscore(normalized_nflips, maxscore, filename = "results/maxscore-vs-nflips.png")