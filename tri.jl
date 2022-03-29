using Triangulate, Plots, TriplotRecipes

"""
    p,t = polytrimesh(pvs, holes, hmax, cmd)

Unstructured meshing of polygonal regions.
    `pvs`: Array of polygons (2xN arrays)
    `holes`: x,y coordinates of holes in geometry (2xM array)
    `hmax`: Scalar or size function (default Inf)
    `cmd`: Command to triangle mesh generator (string, default "puq28.6Q")

    Example:
        pv1 = hcat([0,0], [1,0], [1,1], [.5,.5], [0,1], [0,0])
        pv2 = hcat([.2,.2], [.4,.2], [.4,.4], [.39,.4], [.2,.4], [.2,.2])
        holes = [.3,.3]
        hmaxfcn = (x,y) -> 0.01 + 0.3*abs(sqrt((x-.5)^2+(y-.5)^2))
        p,t = polytrimesh([pv1,pv2], holes, hmaxfcn)
        trimesh(p[1,:], p[2,:], t, aspect_ratio=:equal)
"""
function polytrimesh(pvs, holes=zeros(2,0), hmax=Inf, cmd="puq28.6Q")
  pv = Matrix{Float64}[]
  seg = Matrix{Int32}[]
  nseg = 0
  for cpv in pvs
    closed = false
    if size(cpv,2) > 1 & isapprox(cpv[:,1], cpv[:,end])
      closed = true
      cpv = cpv[:,1:end-1]
    end
    np = size(cpv,2)

    if np > 1
      cseg = [mod(i+j-1,np)+1 for i = 0:1, j = 1:np]
      if !closed
        cseg = cseg[:,1:end-1]
      end

      push!(seg, cseg .+ nseg)
      nseg += size(cseg,2)
    end

    push!(pv, cpv)
  end

  pv = hcat(pv...)
  seg = hcat(seg...)
  triin = Triangulate.TriangulateIO()
  triin.pointlist = pv
  triin.segmentlist = seg
  triin.segmentmarkerlist=Vector{Int32}(1:size(seg,2))
  if holes != nothing
    triin.holelist = reshape(holes, 2, :)
  end

  function unsuitable(x1,y1,x2,y2,x3,y3,area)
    if isa(hmax, Number)
      sz = hmax
    else
      sz = hmax((x1+x2+x3)/3, (y1+y2+y3)/3)
    end
    elemsz = sqrt(maximum([(x1-x2)^2+(y1-y2)^2,
                           (x2-x3)^2+(y2-y3)^2,
                           (x3-x1)^2+(y3-y1)^2]))
    elemsz > sz
  end
  triunsuitable(unsuitable)

  (triout, vorout)=triangulate(cmd, triin)
  triout.pointlist, triout.trianglelist
end

function samplemesh1()
  pv1 = hcat([0,0], [1,0], [1,1], [.5,.5], [0,1], [0,0])
  pv2 = hcat([.2,.2], [.4,.2], [.4,.4], [.39,.4], [.2,.4], [.2,.2])
  holes = [.3,.3]
  hmaxfcn = (x,y) -> 0.01 + 0.3*abs(sqrt((x-.5)^2+(y-.5)^2))
  p,t = polytrimesh([pv1,pv2], holes, hmaxfcn)
  p,t
end

function circlemesh(hmax)
  n = ceil(Int, 2π / hmax)
  θ = 2π*(0:n)'./n
  pv = vcat(cos.(θ), sin.(θ))
  p,t = polytrimesh([pv], [], hmax, "puYQ")
end

#p,t = samplemesh1()
#trimesh(p[1,:], p[2,:], t, aspect_ratio=:equal)

p,t = circlemesh(0.4)
trimesh(p[1,:], p[2,:], t, aspect_ratio=:equal)
