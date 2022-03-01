using EdgeFlip
using MeshPlotter

nref = 1
nflips = 4
env = EdgeFlip.GameEnv(nref,nflips)

fig, ax = MeshPlotter.plot_mesh(env.mesh, d0 = env.d0)
fig.savefig("mesh.png")