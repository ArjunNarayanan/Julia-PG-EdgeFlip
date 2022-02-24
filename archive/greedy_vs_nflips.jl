using EdgeFlip
using MeshPlotter
using Printf
using CSV
using DataFrames

include("greedy_policy.jl")
include("plot.jl")

function render(env)
    fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0)
    return fig
end

function returns_vs_nflips(nref, nflips, num_trajectories; maxstepfactor = 1.2)
    maxsteps = round(Int, maxstepfactor * nflips)
    env = EdgeFlip.GameEnv(nref, nflips)
    avg, dev = GreedyPolicy.mean_and_std_returns(env, maxsteps, num_trajectories)
    
    @printf "NFLIPS = %d \t RET = %1.3f \t DEV = %1.3f\n" nflips avg dev

    return avg, dev
end

num_trajectories = 200
nref = 3
num_edges = EdgeFlip.number_of_edges(EdgeFlip.generate_mesh(nref))
nflips = 1:30:round(Int,1.5num_edges)


returns = [returns_vs_nflips(nref, nf, num_trajectories) for nf in nflips];

avg = [r[1] for r in returns]
dev = [r[2] for r in returns]

filename = "results/greedy-vs-nflips/nref-"*string(nref)*".png"
normalized_nflips = nflips ./ num_edges
# plot_returns(normalized_nflips,avg,dev)
plot_returns(normalized_nflips,avg,dev,filename=filename)


df = DataFrame(:nflips => normalized_nflips, :avg => avg, :dev => dev)
df_filename = "results/greedy-vs-nflips/nref-"*string(nref)*".csv"
CSV.write(df_filename,df)