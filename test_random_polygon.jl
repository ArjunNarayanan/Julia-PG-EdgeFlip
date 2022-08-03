using TriMeshGame
using PyPlot
include("utilities/random_polygon_generator.jl")

TM = TriMeshGame

function plot_polygon(p)
    fig, ax = subplots()
    ax.set_aspect("equal")

    ax.scatter(p[:,1], p[:,2])
    numpts = size(p,1)
    for i = 1:numpts
        pnext = i == numpts ? 1 : i + 1 
        ax.plot([p[i,1],p[pnext,1]],[p[i,2],p[pnext,2]],color="black")
    end
    return fig
end

function mean_vertex_irregularity(d,d0)
    vs = d - d0
    n = count(vs .!= 0)
    return n/length(vs)
end

function delaunay_vertex_irregularity(p)
    pclosed = [p' p[1,:]]
    pout, t = polytrimesh([pclosed], holes=[], cmd="p")
    t = permutedims(t,(2,1))
    angles = polygon_interior_angles(p)
    d0 = desired_valence.(angles)

    mesh = TM.Mesh(p,t)
    return mean_vertex_irregularity(mesh.d, d0)
end

function mean_delaunay_vertex_irregularity(polyorder, numtrials; threshold = 0.1)
    irr = 0.0
    for i = 1:numtrials
        p = random_coordinates(polyorder,threshold=threshold)
        irr += delaunay_vertex_irregularity(p)
    end
    return irr/numtrials
end


irr = mean_delaunay_vertex_irregularity(10, 1000)
