include("../tri.jl")

function circle_env(element_size; maxflipfactor = 1.0)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    num_edges = EdgeFlip.number_of_edges(mesh)
    d0 = fill(6, num_nodes)    

    d0[mesh.bnd_nodes] .= 4
    maxflips = ceil(Int, maxflipfactor*num_edges)

    env = EdgeFlip.GameEnv(mesh, 0, d0 = d0, maxflips = maxflips)

    return env
end

function circle_ordered_game_env(element_size; maxflipfactor = 1.0)
    p, t = circlemesh(element_size)
    mesh = EdgeFlip.Mesh(p, t)
    num_nodes = size(p, 1)
    num_edges = EdgeFlip.number_of_edges(mesh)
    d0 = fill(6, num_nodes)    

    d0[mesh.bnd_nodes] .= 4
    maxflips = ceil(Int, maxflipfactor*num_edges)

    env = EdgeFlip.OrderedGameEnv(mesh, 0, d0 = d0, maxflips = maxflips)

    return env
end