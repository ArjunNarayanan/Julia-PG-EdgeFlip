using LinearAlgebra
include("../tri.jl")

function random_coordinates(n;threshold = 0.2)
    d = range(0,stop=2pi,length=n+1)[1:n]
    r = (1-threshold)*rand(n) .+ threshold

    p = [r .* cos.(d)  r.*sin.(d)]

    return p
end

function enclosed_angle(v1,v2)
    @assert length(v1) == length(v2) == 2
    dotp = dot(v1,v2)
    detp = v1[1]*v2[2] - v1[2]*v2[1]
    rad = atan(detp, dotp)
    if rad < 0
        rad += 2pi
    end

    return rad2deg(rad) 
end

function polygon_interior_angles(p)
    n = size(p,1)
    angles = zeros(n)
    for i = 1:n
        previ = i == 1 ? n : i -1
        nexti = i == n ? 1 : i + 1

        v1 = p[nexti,:] - p[i,:]
        v2 = p[previ,:] - p[i,:]
        angles[i] = enclosed_angle(v1,v2)
    end
    return angles
end

function desired_valence(angle)
    v1 = floor(angle/60)
    v2 = ceil(angle/60)

    e1 = abs(angle - v1*60)
    e2 = abs(angle - v2*60)

    if e1 < e2
        return round(Int,v1) + 1
    else
        return round(Int,v2) + 1
    end
end

function hex_mesh()
    d = range(0,stop=2pi,length=7)[1:end-1]
    p = [cos.(d) sin.(d)]
    t = [1 2 6
         2 5 6
         2 3 5
         3 4 5]
    mesh = TM.Mesh(p, t)
    return mesh
end

function initialize_hex_environment(degree_range,max_actions)
    mesh = hex_mesh()
    wrapper = GameEnvWrapper(mesh, degree_range, max_actions)
    return wrapper
end
