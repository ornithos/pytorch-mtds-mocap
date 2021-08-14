module geom

using LinearAlgebra
using Quaternions
using ArgCheck
using PyCall

pysys = pyimport("sys")

# add python directory to python path if available
pushfirst!(PyVector(pysys."path"), joinpath("."));
isdir(joinpath(pwd(), "pyfiles/")) && pushfirst!(PyVector(pysys."path"), joinpath(pwd(), "pyfiles"));
isdir(joinpath(pwd(), "viz/pyfiles/")) && pushfirst!(PyVector(pysys."path"), joinpath(pwd(), "viz/pyfiles"));

# check we can find the relevant py files.
while !any([isfile(joinpath(p, "Quaternions.py")) for p in pysys."path"])
    @warn "Dan Holden python processing scripts not found."
    println("Please run this julia file from a directory which contains these files,\n" *
             "or contains a 'pyfiles' directory with them.")
    println("\nAlternatively please provide a path on which python can find them:")
    userinput = chomp(readline())
    if isdir(userinput)
        pushfirst!(PyVector(pysys."path"), userinput);
    else
        throw(ErrorException("Cannot find file Quaternions.py"))
    end
end
Quatpy = pyimport("Quaternions");

_rowmaj_reshape_3d(x, ix, iy, iz) = (x=reshape(x, ix, iz, iy); permutedims(x, [1,3,2]);)

# trajectory is represented as (x,z), but most quaternion ops need y (=0) too.
cat_zero_y(x::T, z::T) where T <: Number = [x, 0, z]
cat_zero_y(x::Vector{T}, z::Vector{T}) where T = hcat(x, zeros(T, length(x)), z)
cat_zero_y(X::Matrix{T}) where T = begin; @argcheck size(X,2)==2;
    hcat(X[:,1], zeros(T, size(X,1)), X[:,2]); end

# Quaternion related utils
_qimag = Quaternions.imag
_quat_list(x) = [quat(x[i,:]) for i in 1:size(x,1)]
_quat_list_to_mat(x) = reduce(vcat, [_qimag(xx)' for xx in x])
_quaterion_angle_axis_w_y(θ) = quat(cos(θ/2), 0, sin(θ/2), 0)
_xz_plane_angle(q::Quaternion) = begin; @assert q.v1 ≈ 0 && q.v3 ≈ 0; atan(q.v2, q.s)*2; end  # inverse of above
_apply_rotation(x, qrot) = qrot * x * conj(qrot)
_vec4_to_quat(x::Vector) = begin; @argcheck length(x) == 4; Quaternion(x[1], x[2], x[3], x[4]); end  # quat constructor is WEIRD o.w.
_vec2_to_quat(x::Vector) = begin; @argcheck length(x) == 2;
    Quatpy.Quaternions.between([0,0,1], cat_zero_y(x[1], x[2])).qs[1,:] |> _vec4_to_quat; end



#= ---------------------------------------------------------------------------
          FORWARD KINEMATICS (for Lagrangian skeleton representation)
       These functions used some of Holden's view.py as a starting point.
------------------------------------------------------------------------------=#

reconstruct_modelled(Y::Matrix) = reconstruct(Y, :modelled)
reconstruct_modelled64(Y::Matrix) = reconstruct(Y, :modelled64)
reconstruct_raw(Y::Matrix) = reconstruct(Y, :raw)
reconstruct_root(Y::Matrix) = reconstruct(Y, :root)[:,1,:]

"""
    reconstruct(Y, input_type)

Reconstruct the absolute positions of joints in a contiguous set of frames.
This proceeds by applying forward kinematics using the root rotation from the
Lagrangian representation, and the x-z root velocities of the first few dims of
the processed matrix.

See also shortcuts:
    reconstruct_modelled(Y)
    reconstruct_raw(Y)
    reconstruct_root(Y)
"""
function reconstruct(Y::Matrix{T}, input_type::Symbol) where T <: Number
    Y = convert(Matrix{Float64}, Y)   # reduce error propagation from iterative scheme
    if input_type == :raw
        (size(Y, 2) < 70) && @warn "expecting matrix with >= 70 columns"
        root_r, root_x, root_z, joints = Y[:,1], Y[:,2], Y[:,3], Y[:,8:(63+7)]

    elseif input_type == :modelled64
        (size(Y, 2) != 64) && @warn "expecting matrix with exactly 64 columns"
        N = size(Y, 1)
        root_r, root_x, root_z, joints = Y[:,1], Y[:,2], Y[:,3], Y[:,5:end]
        rootjoint = reduce(hcat,  (zeros(T, N, 1), Y[:,4:4], zeros(T, N, 1)))
        joints = hcat(rootjoint, joints)

    elseif input_type == :modelled
        @assert size(Y, 2) == 67 "expecting matrix with exactly 67 columns. See `reconstruct_modelled64`"
        N = size(Y, 1)
        root_r, root_x, root_z = (Y[:,i] + Y[:,i+3] for i in 1:3)
        rootjoint = reduce(hcat,  (zeros(T, N, 1), Y[:,7:7], zeros(T, N, 1)))
        joints = hcat(rootjoint, Y[:,8:end])

    elseif input_type == :root
        (size(Y, 2) != 3) && @warn "expecting matrix with exactly 3 columns"
        root_r, root_x, root_z = Y[:,1], Y[:,2], Y[:,3]
        joints = zeros(size(Y,1), 3)   # (implicitly starting at the origin.)
    end

    return _joints_fk(joints, root_x, root_z, root_r)
end


function get_traj(Y::Matrix{T}, ixs::AbstractVector; offset_xz=zeros(T, 2),
        offset_r::T=T(0)) where T
    @assert size(Y, 2) == 67 "expecting matrix with exactly 67 columns."
    base_r, base_x, base_z = Y[:,1], Y[:,2], Y[:,3]

    if ixs[1] > 1
        root_r, root_x, root_z = (Y[:,i] + Y[:,i+3] for i in 1:3)

        # Calculate difference between base coods and root over i < ixs[1]
        begin_ix = 1:ixs[1]-1
        start_angle_base = -sum(base_r[begin_ix])
        start_angle_root = -sum(root_r[begin_ix])
        start_angle_δ = let θ=start_angle_base - start_angle_root + offset_r; [sin(θ), cos(θ)]; end

        start_base = _traj_fk(base_x[begin_ix], base_z[begin_ix], base_r[begin_ix],
            start=cat_zero_y(offset_xz...), start_angle=[sin(offset_r), cos(offset_r)])
        start_base = reduce(vcat, v[end] for v in start_base)
        start_root = _traj_fk(root_x[begin_ix], root_z[begin_ix], root_r[begin_ix]) |> x -> reduce(vcat, v[end] for v in x)
        start_δ = cat_zero_y((start_base - start_root + offset_xz)...);

        traj = _traj_fk(base_x[ixs], base_z[ixs], base_r[ixs], start=start_δ ,
                start_angle=start_angle_δ)
    else
        traj = _traj_fk(base_x[ixs], base_z[ixs], base_r[ixs], start=cat_zero_y(offset_xz...),
                start_angle=[sin(offset_r), cos(offset_r)])
    end
    return traj
end

function _traj_fk(root_x::Vector{T}, root_z::Vector{T}, root_r::Vector{T};
        start::Vector=zeros(T, 3), start_angle::Vector=[0,T(1)]) where T <: Number

    n = length(root_x)
    @argcheck norm(start_angle) ≈ 1
    rotation = _vec2_to_quat(start_angle)
    translation = start
    # translation = _qimag(_apply_rotation(quat(start), rotation))  # No. o.w. start pos != start.

    traj = Matrix{T}(undef, 3, n+1);
    traj[1, 1] = translation[1]
    traj[3, 1] = translation[3]

    for i = 1:n
        rotation = _quaterion_angle_axis_w_y(-root_r[i]) * rotation
        next_step = quat(cat_zero_y(root_x[i], root_z[i]))
        translation = translation + _qimag(_apply_rotation(next_step, rotation))
        traj[1, i+1] = translation[1]
        traj[3, i+1] = translation[3]
    end

    return traj[1,:], traj[3,:]
end

function _joints_fk(joints::Matrix{T}, root_x::Vector{T}, root_z::Vector{T},
        root_r::Vector{T}; start::Vector{T}=zeros(T, 3),
        start_angle::Vector{T}=[0,T(1)]) where T <: Number

    n = size(joints, 1)
    njoints = size(joints, 2) ÷ 3
    @assert (njoints * 3 == size(joints, 2)) "number of columns must be div. 3"
    joints = _rowmaj_reshape_3d(joints, n, njoints, 3)

    rotation = _vec2_to_quat(start_angle)
    translation = start

    for i = 1:n
        # Apply the rotation to the skeleton about the y axis
        # so that the azimuth at the **END** of the step is correct.
        joints[i,:,:] = _apply_rotation(_quat_list(joints[i,:,:]), rotation) |> _quat_list_to_mat

        # Move from origin --> calculated next step (see below for def.)
        # Note that this does not happen wrt current azimuth, and so is
        # **INDEPENDENT OF THE ABOVE ROTATION**.
        joints[i,:,1] = joints[i,:,1] .+ translation[1]
        joints[i,:,3] = joints[i,:,3] .+ translation[3]

        # Increment the cumulative y-axis rotation by next rotational difference.
        rotation = _quaterion_angle_axis_w_y(-root_r[i]) * rotation

        # Rotate the next delta (step; x,z diff) and express relative to origin.
        next_step = quat(cat_zero_y(root_x[i], root_z[i]))
        rotated_step = _qimag(_apply_rotation(next_step, rotation))
        translation += rotated_step
    end

    # note that vs the trajectory FK, we don't obtain an (n+1) output. This is
    # because while we know the position of the root at time (n+1), we don't
    # have the relative position of the joints, and so discard this final transformation.
    return joints
end



#= ---------------------------------------------------------------------------
        User facing reconstruction of ground path FK from data matrix
------------------------------------------------------------------------------=#
"""
    fk_path(Y, path_ixs, offset_xzr; for_viz=true, approx_to=nothing)

calculate the path trajectory from a given sequence `Y` (corresponding to an
**entire file**), the the time indices to extract from this sequence `path_ixs`.
The offset_xzr must be specified in order to obtain the initial differences in
position and rotation between the true and smoothed path trajectories. This function
outputs an N x 12 matrix for use in `mocapviz.create_animation` with appropriate
step-ahead entries, but `for_viz=false` will output just the N x 2 path.

To correct the rotation relative to a known trajectory, supply the N x 2 matrix in
`approx_to`, and the `correct_rotation` function will be used to align them.
"""
function fk_path(Y::AbstractMatrix, path_ixs::AbstractVector, offset_xzr::AbstractVector;
        for_viz::Bool=true, approx_to::Union{Nothing, AbstractMatrix}=nothing)
    N = length(path_ixs)
    T = promote_type(eltype(Y), eltype(offset_xzr))
    Y, offset_xzr = T.(Y), T.(offset_xzr)
    _traj = hcat(get_traj(Y, path_ixs, offset_xz=offset_xzr[1:2], offset_r=offset_xzr[3])...)

    if !(approx_to === nothing)
        _traj = _optimise_start(_traj, approx_to, vcat(zeros(T, 2), [randn(T, 2)*10 for i in 1:50]...))
    end
    !for_viz && return _traj

    path = randn(N, 12)*0.01
    for i in 1:6, j in 0:1
        path[:,i + j*6] = vcat(_traj[1+(5*i):N,1+j], _traj[N,1+j] .+ randn(5*i)*0.01)
    end
    return path
end

"""
    correct_rotation(A, B)
    correct_rotation(approx, target)

rotate path matrix A (n x 2) to closest match with target path
matrix B (n x 2) using orthogonal procrustes. This function
returns the corrected path matrix Â.
"""
function correct_rotation(A, B)
    @argcheck size(A, 2) == size(B, 2) == 2
    prod_svd = svd(B' * A)
    return A * prod_svd.V * prod_svd.U'
end

function _procrustes_matrix(A, B)
    prod_svd = svd(B' * A)
    return prod_svd.V * prod_svd.U'
end

function _optimise_start(A, B, xz::Vector)
    err = [sum(x->x^2, A[1:end-1,:] * _procrustes_matrix(A[1:end-1,:] .+ pos', B) - B) for pos in xz]
    return A * (_procrustes_matrix(A[1:end-1,:] .+ xz[argmin(err)]', B))
end
end