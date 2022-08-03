include("random_polygon_generator.jl")


angles = 1:360
v = desired_valence2.(angles)
dv = diff(v)
idx = findall(dv .!= 0)

change_angles = idx .+ 1