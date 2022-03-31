using EdgeFlip
using MeshPlotter
include("tri.jl")
include("greedy_policy.jl")

GP = GreedyPolicy

function normalized_returns_vs_polygon_degree(element_size)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    num_edges = EdgeFlip.number_of_edges(mesh)
    d0 = fill(6, num_nodes)
    d0[mesh.bnd_nodes] .= 4

    polygon_degree = length(mesh.bnd_nodes)
    maxflips = num_edges
    env = EdgeFlip.GameEnv(mesh, 0, d0 = d0, fixed_reset = true, maxflips = num_edges)
    optimum_score = env.score - abs(sum(env.vertex_score))

    num_trajectories = 500
    ret = GP.average_returns(env, num_trajectories)
    normalized_return = optimum_score == 0.0 ? 1.0 : ret / optimum_score

    return polygon_degree, num_nodes, num_edges, optimum_score, ret, normalized_return
end

function optimum_score_vs_polygon_degree(element_size)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    d0 = fill(6, num_nodes)
    d0[mesh.bnd_nodes] .= 4

    polygon_degree = length(mesh.bnd_nodes)

    vs = mesh.d - d0
    opt_score = abs(sum(vs))

    return polygon_degree, opt_score
end


element_size = 0.3
p, t = circlemesh(element_size)
mesh = EdgeFlip.Mesh(p, t)
num_nodes = size(p, 1)
num_edges = EdgeFlip.number_of_edges(mesh)
d0 = fill(6, num_nodes)
d0[mesh.bnd_nodes] .= 4

polygon_degree = length(mesh.bnd_nodes)
maxflips = ceil(Int,2num_edges)
env = EdgeFlip.GameEnv(mesh, 0, d0 = d0, fixed_reset = true, maxflips = num_edges)
optimum_score = env.score - abs(sum(env.vertex_score))

num_trajectories = 500
ret = GP.average_returns(env, num_trajectories)
normalized_return = optimum_score == 0.0 ? 1.0 : ret / optimum_score











# element_size = [0.8,0.7,0.6,0.5,0.4,0.3]
# vals = normalized_returns_vs_polygon_degree.(element_size)

# cols = [[v[i] for v in vals] for i in 1:6]


# using CSV, DataFrames
# df = DataFrame("polygon degree" => cols[1],
#                "num nodes" => cols[2],
#                "num edges" => cols[3],
#                "opt ret" => cols[4],
#                "avg ret" => cols[5],
#                "normalized avg ret" => cols[6])

# using PyPlot
# fig,ax = subplots()
# ax.plot(df[!, "polygon degree"], df[!, "normalized avg ret"])
# fig


# filename = "results/circlemesh/return-stats.csv"
# CSV.write(filename, df)