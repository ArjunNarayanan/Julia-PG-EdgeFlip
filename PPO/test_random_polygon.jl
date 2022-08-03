include("random_polygon_generator.jl")

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

p = random_coordinates(10,threshold=0.1)
plot_polygon(p)

