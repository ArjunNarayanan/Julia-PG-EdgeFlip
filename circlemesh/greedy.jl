using EdgeFlip
using MeshPlotter
include("../greedy_policy.jl")
include("circlemesh.jl")

EF = EdgeFlip
GP = GreedyPolicy
MP = MeshPlotter

function return_trajectory(env)
    done = EF.is_terminated(env)
    ret = zeros(Int,env.maxflips)
    counter = 1

    while !done
        action = GP.greedy_action(env)
        EF.step!(env, action)

        ret[counter] = EF.reward(env)
        counter += 1

        done = EF.is_terminated(env)
    end

    return ret
end


element_size = 0.1
maxflips = 1000
env = circle_env(element_size, maxflips)

EF.reset!(env)
maxreturn = env.score - env.optimum_score

# EF.averagesmoothing!(env.mesh, 10)
# fig, ax = MP.plot_mesh(env.mesh, d0 = env.d0, vertex_score = false)
# fig
# fig.savefig("circlemesh/original-0-1.png")

EF.reset!(env)
ret = return_trajectory(env)
curet = cumsum(ret)

# using PyPlot
# fig, ax = subplots()
# ax.plot(curet)
# ax.set_ylim(0,maxreturn)
# fig

EF.averagesmoothing!(env.mesh,10)
fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0, vertex_score = false)
fig
fig.savefig("circlemesh/best-greedy-0-1-no-score.png")