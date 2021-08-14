module mocaputil

using StatsBase, Statistics, ArgCheck
import Random: shuffle

export transform, scale_transform
export fit
export invert
export MyStandardScaler

"""
    fit(MyStandardScaler, X, dims)

Fit a standardisation to a matrix `X` s.t. that the rows/columns have mean zero and standard deviation 1.
This operation calculates the mean and standard deviation and outputs a MyStandardScaler object `s` which
allows this same standardisation to be fit to any matrix using `transform(s, Y)` for some matrix `Y`. Note
that the input matrix `X` is *not* transformed by this operation. Instead use the above `transform` syntax
on `X`.

Note a couple of addendums:

1. Any columns/rows with constant values will result in a standard deviation of 1.0, not 0. This is to avoid NaN errors from the transformation (and it is a natural choice).
2. If only a subset of the rows/columns should be standardised, an additional argument of the indices may be given as:

    `fit(MyStandardScaler, X, operate_on, dims)`

    The subset to operate upon will be maintained through all `transform` and `invert` operations.
"""
mutable struct MyStandardScaler{T}
    μ::Array{T,1}
    σ::Array{T,1}
    operate_on::Array{L where L <: Int,1}
    dims::Int
end

Base.copy(s::MyStandardScaler) = MyStandardScaler(copy(s.μ), copy(s.σ), copy(s.operate_on), copy(s.dims))

StatsBase.fit(::Type{MyStandardScaler}, X::Matrix, dims::Int=1) = MyStandardScaler(vec(mean(X, dims=dims)),
    vec(std(X, dims=dims)), collect(1:size(X,3-dims)), dims) |> post_fit

StatsBase.fit(::Type{MyStandardScaler}, X::Matrix, operate_on::Array{L where L <: Int}, dims::Int=1) =
    MyStandardScaler(vec(mean(X, dims=dims))[operate_on],
    vec(std(X, dims=dims))[operate_on],
    operate_on, dims) |> post_fit

function post_fit(s::MyStandardScaler)
    bad_ixs = (s.σ .== 0)
    s.σ[bad_ixs] .= 1
    s
end

function scale_transform(s::MyStandardScaler, X::Matrix, dims::Int=s.dims)
    (dims != s.dims) && @warn "dim specified in transform is different to specification during fit."
    tr = dims == 1 ? transpose : identity
    tr2 = dims == 2 ? transpose : identity
    if s.operate_on == 1:size(X, 3-dims)
        out = (X .- tr(s.μ)) ./ tr(s.σ)
    else
        out = tr2(copy(X))
        s_sub = copy(s)
        s_sub.operate_on = 1:length(s.operate_on)
        s_sub.dims = 1
        out[:, s.operate_on] = transform(s_sub, out[:, s.operate_on], 1)
        out = tr2(out)
    end
    out
end

function invert(s::MyStandardScaler, X::Matrix, dims::Int=s.dims)
    (dims != s.dims) && @warn "dim specified in inversion is different to specification during fit."
    tr = dims == 1 ? transpose : identity
    tr2 = dims == 2 ? transpose : identity
    if s.operate_on == 1:size(X, 3-dims)
        out = (X .* tr(s.σ)) .+ tr(s.μ)
    else
        out = tr2(copy(X))
        s_sub = copy(s)
        s_sub.operate_on = 1:length(s.operate_on)
        s_sub.dims = 1
        out[:, s.operate_on] = invert(s_sub, out[:, s.operate_on], 1)
        out = tr2(out)
    end
    out
end

"""
DataIterator is conceptually similar to a DataLoader class in PyTorch. It has the following methods:

* `iterate`   (obtain the next iteration of data)
* `weights`   (extract all weights from the DataIterator for each datapoint and normalize)
* `copy`
* `shuffle`   (shuffles the order of the data)
* `indexed_shuffle`  (shuffles the order of the data, and additionally spits out the indices of the new order)
"""
struct DataIterator
    data::Array{D}  where {D <: Dict}
    batch_size::Int
    min_size::Int
    start::Int
    DataIterator(d, b_sz, m_sz, start) = begin; @assert m_sz < b_sz; new(d, b_sz, m_sz, start); end
end
DataIterator(data, batch_size; min_size=1, start=0) = DataIterator(data, batch_size, min_size, start)
Base.copy(di::DataIterator) = DataIterator(deepcopy(di.data), copy(di.batch_size), copy(di.min_size), copy(di.start))

function Base.iterate(iter::DataIterator, state=(1, iter.start))
    element, ix = state

    (element > length(iter.data)) && return nothing
    while ix + iter.min_size > size(iter.data[element][:Y], 2)
        element += 1
        ix = iter.start
        (element > length(iter.data)) && return nothing
    end
    new_state = ix == iter.start   # not in while, since 1st iterate in general won't use this loop.

    chunk  = iter.data[element]
    cur_length = size(chunk[:Y], 2)
    eix  = min(ix + iter.batch_size, cur_length)
    ix += 1

    return ((chunk[:Y][:,ix:eix], chunk[:U][:,ix:eix], new_state), (element, eix))
end

function weights(iter::DataIterator; as_pct::Bool=true)
    w = [size(y,2) for (y, u, new_state) in iter]
    return as_pct ? w / sum(w) : w
end

function Base.length(iter::DataIterator)
    map(iter.data) do x
        d, r = divrem(size(x[:Y],2)-iter.start, iter.batch_size);
        d + (r >= iter.min_size);
        end |> sum
end

function shuffle(di::DataIterator)
    newdata = shuffle([Dict(:Y=>cY, :U=>cU) for (cY, cU, h0) in di])
    DataIterator(newdata, di.batch_size, 1, di.start)
end


function indexed_shuffle(di::DataIterator)
    order = shuffle(1:length(di))
    dataArray = [Dict(:Y=>cY, :U=>cU) for (cY, cU, h0) in di]
    newdata = [dataArray[i] for i in order]
    order, DataIterator(newdata, di.batch_size, 1, di.start)
end


function get_file_pos_from_iter(di::DataIterator, styles_lkp::Vector,
        style_ix::Int, ix::Int)
    ffs = vcat([s for (i,s) in enumerate(styles_lkp) if i != style_ix]...)
    e, file_offset = 1, di.start
    for i in 1:ix-1
        _, (e, file_offset) = iterate(di, (e, file_offset))
    end
    return ffs[e], file_offset+1
end

function get_file_pos_from_iter_test(di::DataIterator, styles_lkp::Vector,
        style_ix::Int, ix::Int)
    get_file_pos_from_iter(di, [styles_lkp[style_ix]], 0,ix)
end

end