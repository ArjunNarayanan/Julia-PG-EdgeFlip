using EdgeFlip
using MeshPlotter
using Distributions: Categorical
include("../NL_policy.jl")
include("circlemesh.jl")

EF = EdgeFlip
MP = MeshPlotter

function state(env)
    ets = copy(EdgeFlip.edge_template_score(env))
    econn = copy(EdgeFlip.edge_connectivity(env))
    epairs = copy(EdgeFlip.edge_pairs(env))

    idx = findall(epairs .== 0)
    epairs[idx] .= idx

    return ets, econn, epairs
end

function select_action(env, policy)
    p = softmax(vec(PG.eval_single(policy, state(env)...)))
    idx = rand(Categorical(p))
    action = PG.idx_to_action(idx)
    return action
end

function return_trajectory(env, policy)
    done = EF.is_terminated(env)
    ret = zeros(Int,env.maxflips)
    counter = 1

    while !done
        tri, ver = select_action(env, policy)
        EF.step!(env, tri, ver)

        ret[counter] = EF.reward(env)
        counter += 1

        done = EF.is_terminated(env)
    end

    return ret
end

element_size = 0.1
maxflips = 500
env = circle_ordered_game_env(element_size, maxflips)

# using BSON
# data = BSON.load("results/models/5L-model/policy-63500.bson")
# policy = data[:policy]

# EF.reset!(env)
# maxreturn = env.score - env.optimum_score

EF.reset!(env)
ret = return_trajectory(env, policy)
curet = cumsum(ret)
maximum(curet)


EF.averagesmoothing!(env.mesh,10)
fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0, vertex_score = false)
fig
fig.savefig("circlemesh/best-3L-network-0-1.png")