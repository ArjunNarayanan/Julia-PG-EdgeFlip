using PyPlot
using Statistics
using EdgeFlip
include("greedy_policy.jl")
include("utilities/random_polygon_generator.jl")
GP = GreedyPolicy

function random_polygon_mesh(polyorder, threshold)
    p = random_coordinates(polyorder,threshold=threshold)
    pclosed = [p' p[1,:]] 
    pout, t = polytrimesh([pclosed], holes=[], cmd="p")
    t = permutedims(t,(2,1))
    mesh = EdgeFlip.Mesh(p, t)
    return mesh
end

function random_polygon_env(polyorder, maxflips, threshold)
    mesh = random_polygon_mesh(polyorder, threshold)
    d0 = desired_valence.(polygon_interior_angles(mesh.p))
    env = EdgeFlip.GameEnv(mesh, 0, d0 = d0, maxflips = maxflips)    
end

function average_greedy_returns(polyorder, maxflips, numtrials; threshold = 0.1)
    ret = zeros(numtrials)
    for trial in 1:numtrials
        env = random_polygon_env(polyorder, maxflips, threshold)
        ret[trial] = GP.single_trajectory_normalized_return(env)
    end
    return mean(ret), std(ret)
end

function plot_returns(maxflips, ret, dev; filename = "", ylim = (0.0, 1.0))
    fig, ax = subplots()
    ax.plot(maxflips, ret)
    ax.fill_between(maxflips, ret + dev/2, ret - dev/2, alpha = 0.2, facecolor = "blue")
    ax.set_xlabel("Maximum allowed flips")
    ax.set_ylabel("Normalized returns")
    ax.set_ylim(ylim)
    if length(filename) > 0
        fig.savefig(filename)
    end
    return fig
end

function actions_and_scores_history(env)
    actions = []
    scores = []
    done = EdgeFlip.is_terminated(env)
    
    while !done
        action = GP.greedy_action(env)
        EdgeFlip.step!(env, action)

        push!(actions, action)
        push!(scores, EdgeFlip.score(env))

        done = EdgeFlip.is_terminated(env)
    end
    return actions, scores
end

function best_greedy_mesh!(env)
    EdgeFlip.reset!(env)
    actions, scores = actions_and_scores_history(env)
    idx = argmin(scores)
    EdgeFlip.reset!(env)
    for a in actions[1:idx]
        EdgeFlip.step!(env, a)
    end
end

polyorder = 10
threshold = 0.1


# maxflip_range = 0:50
# numtrials = 1000

# stats = [average_greedy_returns(polyorder, mf, numtrials) for mf in maxflip_range]
# ret = [first(s) for s in stats]
# dev = [last(s) for s in stats]

# fig = plot_returns(maxflip_range, ret, dev)
# filename = "results/greedy-random-polygon/returns-vs-maxflips/polyorder"*string(polyorder)*"-returns.png"
# fig.savefig(filename)





using MeshPlotter
counter = 0

env = random_polygon_env(10, 20, threshold)
ret = GP.normalized_returns(env, 1000)

counter += 1
EdgeFlip.reset!(env)
fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0)
fig
filename = "results/greedy-random-polygon/improved-meshes/initial-"*string(counter)*".png"
fig.savefig(filename)

best_greedy_mesh!(env)
fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0)
fig
filename = "results/greedy-random-polygon/improved-meshes/improved-"*string(counter)*".png"
fig.savefig(filename)






