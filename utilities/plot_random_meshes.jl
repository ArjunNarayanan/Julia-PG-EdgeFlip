using EdgeFlip
using MeshPlotter
include("random_polygon_generator.jl")

function random_polygon_mesh(polyorder, threshold)
    p = random_coordinates(polyorder,threshold=threshold)
    pclosed = [p' p[1,:]] 
    pout, t = polytrimesh([pclosed], holes=[], cmd="p")
    t = permutedims(t,(2,1))
    mesh = EdgeFlip.Mesh(p, t)
    return mesh
end

function generate_random_mesh_plots(polyorder, threshold, num_meshes)
    for i in 1:num_meshes
        mesh = random_polygon_mesh(polyorder, threshold)
        d0 = desired_valence.(polygon_interior_angles(mesh.p))
        fig, ax = MeshPlotter.plot_mesh(mesh, d0 = d0)
        filename = "utilities/figures/mesh"*string(i)*".png"
        fig.tight_layout()
        fig.savefig(filename)
    end
end

generate_random_mesh_plots(10, 0.1, 10)