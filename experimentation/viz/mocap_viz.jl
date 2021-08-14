module mocapviz

using Quaternions
using MeshCat
using CoordinateTransformations: LinearMap, AffineMap, Translation
using Rotations: rotation_between
using GeometryBasics: Cylinder, HyperSphere, Point
using LinearAlgebra: norm, diagm
using DSP   # conv
using Colors   # RGB, RGBA, colorant_str
using Formatting, ArgCheck

import MeshCat: GenericMaterial

Base.convert(::Type{Tuple}, c::RGB) = (c.r, c.g, c.b)
transparency(c::RGB, α) = RGBA(convert(Tuple, c)..., α)
darkencol(x::RGB) = whitebalance(x, colorant"white", colorant"grey55")
darkencol(x::RGBA) = let u=convert(RGB, x); y=darkencol(u); y=RGBA(y.r, y.g, y.b, x.alpha); y; end
darkencol(x::GenericMaterial) = GenericMaterial(_type=x._type, color=darkencol(x.color))   # includes MeshPhongMaterial

const blackmesh = MeshPhongMaterial(color=RGBA(0.1, 0.1, 0.1, 0.4))
const greymesh = MeshPhongMaterial(color=RGBA(0.3, 0.3, 0.3, 0.7))
const yellowmesh = MeshPhongMaterial(color=RGBA(204/255, 204/255, 0., 0.7))
const redmesh = MeshPhongMaterial(color=RGBA(224/255, 131/255, 94/255, 0.7))

const bluemesh = MeshPhongMaterial(color=transparency(colorant"#4C9DFF", 0.8));
const greenmesh = MeshPhongMaterial(color=transparency(colorant"#4CE54C", 0.8));
const deepredmesh = MeshPhongMaterial(color=transparency(colorant"#E56666", 0.8));

# note that the y-axis (vertical) in three.js is the *last*/*3rd* dimension.
project_ground!(x::AbstractVector) = begin; (length(x)==0 && return); @assert length(x) == 3; x[3] = 0; end;
project_ground(x::AbstractVector) = begin; x = copy(x); project_ground!(x); x; end
project_ground!(X::AbstractMatrix) = begin; (length(X)==0 && return); @assert size(X,2) == 3; X[:,3] .= 0; end;
project_ground(X::AbstractMatrix) = begin; X = copy(X); project_ground!(X); X; end


# initialise lines and dots
mutable struct LineCollection{T, R}
    data::Array{T}
    vnames::Array{String}
    attributes::Array{R}
end
mutable struct JointCollection{T, R}
    data::Array{T}
    vnames::Array{String}
    attributes::Array{R}
end

Base.length(x::LineCollection) = length(x.data)
Base.length(x::JointCollection) = length(x.data)

function update_pos!(lines::LineCollection, origin::Matrix, extremity::Matrix)
    @argcheck length(lines) == size(origin, 1) == size(extremity, 1)
    lines.data = [Cylinder(Point{3}(origin[i,:]), Point{3}(extremity[i,:]), l.r)
                  for (i,l) in enumerate(lines.data)]
end

function update_pos!(lines::LineCollection, data::Matrix, parents::Vector)
    parents = parents[1:length(lines)]
    parent = data[parents,:]
    data = data[2:end, :]  # remove root (has no line)
    update_pos!(lines, parent, data)
end

function update_pos!(joints::JointCollection, data::Matrix)
    @argcheck length(joints) == size(data, 1)
    joints.data = [HyperSphere(Point{3}(data[i,:]), l.r) for (i,l) in enumerate(joints.data)]
end

# 3D plotting utilities: map line from x₁ → x₂ into affine transformation of e₁.
function line_into_transform(x1, x2)
    scaling, position, rotation = line_into_transforms_indiv(x1, x2)
    return AffineMap(rotation, position) ∘ LinearMap(diagm(0=>scaling))
end

function line_into_transforms_indiv(x1, x2)
    scaling = [norm(x2-x1), 1., 1]
    position = x1
    rotation = rotation_between([1., 0, 0], x2 - x1)
    return scaling, position, rotation
end

line_into_transform(x::Cylinder) = line_into_transform(x.origin, x.extremity)
line_into_transforms_indiv(x::Cylinder) = line_into_transforms_indiv(x.origin, x.extremity)

function setobj_collection!(v::AbstractVisualizer, args...)
    for arg in args
        setobj_collection!(v, arg)
    end
end

function setobj_collection!(v::AbstractVisualizer, coll::Union{LineCollection,JointCollection})
    for i in 1:length(coll)
        setobject!(v[coll.vnames[i]], coll.data[i], coll.attributes[i])
    end
end


function settransform_collection!(v::AbstractVisualizer, args...; anim::Bool=false)
    for arg in args
        settransform_collection!(v, arg; anim=anim)
    end
end

function settransform_collection!(v::AbstractVisualizer, L::LineCollection; anim::Bool=false)
    if false #anim
        # for i in 1:length(L)
        #     scaling, position, rotation = line_into_transforms_indiv(L.data[i])
        #     anim_settransform!(v[L.vnames[i]], scaling, position, rotation)
        # end
    else
        for i in 1:length(L)
            tform = line_into_transform(L.data[i])
            settransform!(v[L.vnames[i]], tform)
        end
    end
end

function settransform_collection!(v::AbstractVisualizer, J::JointCollection; anim::Bool=false)
    for i in 1:length(J)
        settransform!(v[J.vnames[i]], Translation(J.data[i].center))
    end
end


# # HACK: alternative method for rotation∘scale because MeshCat *animation* lowering is broken
# # -> This was used for a previous version of MeshCat. I don't think it's needed anymore, HOWEVER, the angles
# # between joints is a little off. I don't have time to chase this down.
# function anim_settransform!(vis::MeshCat.AbstractVisualizer, scaling, position, rotation)
#     clip = MeshCat.getclip!(vis)
#     MeshCat._setprop!(clip, vis.current_frame, "scale", "vector3", scaling)
#     MeshCat._setprop!(clip, vis.current_frame, "position", "vector3", position)
#     MeshCat._setprop!(clip, vis.current_frame, "quaternion", "quaternion", MeshCat.js_quaternion(rotation))
# end


function create_animation(data::Vector, names::Union{String, Array{String}}="dataset";
    vis=nothing, parents=[1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20],
    jointmesh::Union{AbstractMaterial, Vector{T} where T <: AbstractMaterial}=greymesh,
    linemesh::Union{AbstractMaterial, Vector{T} where T <: AbstractMaterial}=yellowmesh, scale=0.1,
    camera::Symbol=:front, path::Union{Nothing, AbstractMatrix{T} where T <: Real, Vector}=nothing)

    # dimensions
    Ts = [size(d,1) for d in data]
    ls = [size(d,2) for d in data]
    n  = length(ls)
    maxT = maximum(Ts)
    ispath = !(path === nothing)

    # convenience expansions if args are singular
    (names isa String) && (names = [names * string(i) for i in 1:n])
    (jointmesh isa AbstractMaterial) && (jointmesh = repeat([jointmesh], n))
    (linemesh isa AbstractMaterial) && (linemesh = repeat([linemesh], n))
    ispath && (path isa AbstractMatrix) && (path = [path])
    npaths = ispath ? length(path) : 0
    (npaths >= 1) && (pathmesh = map(darkencol, linemesh));
    (npaths == 1) && (pathmesh = [blackmesh]);
    # assertions
    @argcheck length(data) == length(names)
    @assert (all([ndims(d) for d in data] .== 3) && all([size(d,3) for d in data] .==3)) "Need NxJx3 matrices."
    @argcheck camera in [:front, :back, :static]
    ispath && @argcheck size(path[1]) == (Ts[1], 12)
    # @assert all(ls .<= length(parents)-1) "more joints given (dim 1) than are assigned parents."
    # any(ls .> length(parents)) && @warn format("({:d}/{:d}) datasets have fewer joints than specified in parents.",
    #     sum(ls .> length(parents)), n)

    if vis === nothing
        vis = Visualizer()
        open(vis)
    else
        @assert (vis isa Visualizer) "Don't know what to do with vis::$(typeof(vis)). Expecting Visualizer."
    end

    gen_numstr(prefix) = num -> prefix * format("{:02d}", num);
    dotstr, linestr, pathstr = gen_numstr("dot_"), gen_numstr("line_"), gen_numstr("path_")

    # Initialise objects for each dataset
    objs = map(1:n) do i
        if i <= npaths
            obj_path = LineCollection(repeat([Cylinder(zero(Point{3, Float64}), Point{3}([1.,0,0]), 0.02)], 10), map(pathstr,  1:10), repeat([pathmesh[i]], 10))
            setobj_collection!(vis[names[i]], obj_path)
        else
            obj_path = nothing
        end

        nj, nl = ls[i], ls[i]-1
        obj_lc = LineCollection(repeat([Cylinder(zero(Point{3, Float64}), Point{3}([1.,0,0]), 0.03)], nl), map(linestr, 1:nl), repeat([linemesh[i]], nl))
        obj_jc = JointCollection(repeat([HyperSphere(Point{3}(0.), 0.04)], nj), map(dotstr, 1:nj), repeat([jointmesh[i]], nj))
        setobj_collection!(vis[names[i]], obj_lc, obj_jc)  # note order: lines at back, joints at front
        (obj_lc, obj_jc, obj_path)
    end

    #=========== BEGIN ANIMATION CONSTRUCTION ===============================#
    anim = Animation()

    # camera trajectory (smoothed (MA-4) over skeleton traj)
    cam_ix = 1
    pos = hcat([data[cam_ix][tt,1,1]*scale for tt in 1:maxT], [data[cam_ix][tt,1,3]*scale for tt in 1:maxT])
    pos = hcat(conv(pos[:,1], Windows.rect(4)/4), conv(pos[:,2], Windows.rect(4)/4))[1:maxT,:]
    cc = [scale, scale] * (camera == :front ? -1 : +1)

    reord = [1,3,2] # need to permute z/y axis for three.js setup
    for tt in 1:maxT
        atframe(anim, tt-1) do
            for i in 1:n
                if Ts[i] >= tt
                    lines, joints, pathlines = objs[i]
                    update_pos!(lines, data[i][tt,:,reord]*scale, parents)   # Note that update_pos only updates the line/joint collection position.
                    update_pos!(joints, data[i][tt,:,reord]*scale);          # Once these objects are correct, `settransform_collection` pushes => animation.
                    settransform_collection!(vis[names[i]], lines, joints; anim=true)  # note order: lines at back, joints at front

                    if i <= npaths
                        trail = randn(5,3)*0.001   # case i=1: all segments need positive length (Meshcat reqmt.)
                        trail_ixs = max(tt-41, 1):10:(tt-1)
                        if false
                            trail[(6-length(trail_ixs)):5, :] = project_ground(data[i][trail_ixs, 1, reord])
                        else
                            trail[(6-length(trail_ixs)):5, :] = hcat(path[i][trail_ixs, 1], path[i][trail_ixs, 7], zeros(length(trail_ixs)))
                        end
                        path_matrix = vcat(trail, hcat(reshape(path[i][tt, :], 6, 2), zeros(6)))*scale
                        update_pos!(pathlines, path_matrix[1:10,:], path_matrix[2:11,:])
                        settransform_collection!(vis[names[i]], pathlines; anim=true)
                    end
                end
            end
            # one-per-frame updates
            if camera != :static
                settransform!(vis["/Cameras/default"], Translation(pos[tt,1] + cc[1], pos[tt,2] + cc[2], 1))
            end
        end
    end

    setanimation!(vis, anim)
    return vis
end


function plot_2d_skeleton(joints::AbstractMatrix)
    @argcheck size(joints) == (21, 3)
    parents=[1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]
    scatter(joints[:,1], joints[:,2]);
    [plot([[joints[p,j], joints[e+1,j]] for j in 1:2]..., c="k") for (e,p) in enumerate(parents)]
    [text(joints[i,1]+0.2, joints[i,2]-0.7, i) for i in 1:21]
    gca().set_aspect("equal")
    gca().axis("off");
end

function plot_2d_skeleton(root_y::AbstractFloat, joints::AbstractMatrix)
    @argcheck size(joints) == (20, 3)
    joints = vcat([0 root_y 0], joints);
    plot_2d_skeleton(joints)
end

end
