module expmtdata
using ArgCheck

mutable struct ExperimentData{MT <: AbstractMatrix}
    YsRaw::Vector{MT}
    Ys::Vector{MT}
    Us::Vector{MT}
    ix_lkp
    function ExperimentData(YsRaw, Ys, Us, ix_lkp)
        MT = unique([typeof(x) for x in YsRaw])[1]
        YsRaw = [y[2:end,:] for y in YsRaw]  # note that Y, U take 2:end, 1:end-1.
        YsRaw = convert(Vector{MT}, YsRaw)
        Ys = convert(Vector{MT}, Ys)
        Us = convert(Vector{MT}, Us)
        new{MT}(YsRaw, Ys, Us, ix_lkp)
    end
end


"""
    get_data(s::ExperimentData, ix, splittype, tasktype)

Convenience utility for accessing data stored in an ExperimentData struct.
Specify the index of the target task, and then select from:

splittype:
* **:all**        - return the concatentation of all training/validation/test data.
* **:trainvalid** - return the concatentation of all training/validation data.
* **:split**      - return individual (3x) outputs for training/validation/test data.
* **:test**       - return only the test data
* **:train**      - return only the train data
* **:valid**      - return only the validation data.

tasktype:
* **:stl**   - single task model. Return train/validation/test data from this task's data.
* **:pool**  - pooled/cohort model. Here, training and validation data are from the
         complement of the selected index, returned in individual wrappers.

Note that in all cases, the output will be (a) Dict(s) containing the following fields:

* **:Y**    - the observation matrix (each column is an observation).
* **:U**    - the input matrix (each column is a datapoint).
* **:Yraw** - the raw data before standardisation and another manipulation. (Possibly legacy?)

Othe kwargs:

* `concat`  - By default, each boundary encountered between files will result in
a separate Dict, so the return values will be a vector of Dicts. However, for
more basic models (such as linear regression) with no assumption of temporal
continuity, it may be simpler to operate on a standard input and output data
matrix. Setting `concat = true` will return just a single Dict in an array with
all the data. Choosing `simplify=true` will further remove the array, returning
only Dicts.
* `stratified`  - (STL only) stratify the validation/test sets across files in
each style. By default, the test set will come at the end of the concatenation
of all files. Stratifying will mean there are L test sets from each of L files.
For the pooled dataset, the test set is partially stratified, that is, it is
stratified over the *types* (i.e. a % of each style), but not over the *files*
within the types. Given that our goal is MTL, this seems appropriate.
* `split`  - The train/validation/test split as a simplicial 3-dim vector.
* `simplify` - See `concat`. Used without `concat` this option does nothing.
"""
function get_data(s::ExperimentData, ix::Int, splittype::Symbol, tasktype::Symbol;
        concat::Bool=false, stratified=false, split=[0.7,0.1,0.2], simplify::Bool=false)
    @argcheck splittype ∈ [:all, :trainvalid, :split, :test, :train, :valid]
    @argcheck tasktype ∈ [:stl, :pool, :pooled]
    @assert !(stratified && splittype != :stl) "stratified only available for STL. Pooled is 'semi-stratified'(!)"
    if tasktype == :stl
        get_data_stl(s, ix, splittype; concat=concat, stratified=stratified,
            split=split, simplify=simplify)
    else
        splitpool = split[1:2]./sum(split[1:2])
        get_data_pooled(s, ix, splittype; concat=concat, split=splitpool, simplify=simplify)
    end
end


function get_data_stl(s::ExperimentData, ix::Int, splittype::Symbol;
        concat::Bool=false, stratified=false, split=[0.7,0.1,0.2], simplify=false)
    @argcheck splittype ∈ [:all, :trainvalid, :split, :test, :train, :valid]
    ixs = s.ix_lkp[ix]
    stratified = false

    # Get STL data (needed for everything)
    cYsraw = s.YsRaw[ixs]
    cYs    = s.Ys[ixs]
    cUs    = s.Us[ixs]

    if splittype == :all
        if concat
            cYsraw, cYs, cUs = _concat(cYsraw, cYs, cUs, simplify);
        end
        return _create_y_u_raw_dict(cYs, cUs, cYsraw)
    end

    Ns     = [size(y, 2) for y in cYs]
    !stratified && begin; cYsraw, cYs, cUs = _concat(cYsraw, cYs, cUs, false); end

    train, valid, test = create_data_split(cYs, cUs, cYsraw, split);
    if !stratified && !concat
        train, valid, test = _unconcatDicts(train, valid, test, Ns)
    elseif concat
        train, valid, test = _concatDicts(train, valid, test)
        if simplify
            rmv(x) = (length(x) == 1) ? x[1] : x
            train, valid, test = rmv(train), rmv(valid), rmv(test)
        end
    end

    if splittype == :split
        return train, valid, test
    elseif splittype == :test
        return test
    elseif splittype == :train
        return train
    elseif splittype == :valid
        return valid
    elseif splittype == :trainvalid
        return _concatDict([train, valid])
    else
        error("Unreachable error")
    end
end


function get_data_pooled(s::ExperimentData, ix::Int, splittype::Symbol;
        concat::Bool=false, split=[0.875, 0.125], simplify=false)
    @argcheck splittype ∈ [:all, :split, :test, :train, :valid]
    @argcheck length(split) == 2

    if splittype == :all
        L = length(s.ix_lkp)
        all = vcat([get_data_stl(s, i, :all; concat=concat) for i in 1:L]...)
        rmv = !simplify ? identity : (x-> (length(x) == 1) ? x[1] : x)
        return !concat ? all : _concatDicts(all)
    end

    # Get STL data (needed for everything)
    test = get_data_stl(s, ix, :all; concat=concat, stratified=false, simplify=simplify)

    if splittype == :test
        return test
    end

    train = Dict[]
    valid = Dict[]

    for i in setdiff(1:length(s.ix_lkp), ix)
        _train, _valid, _test = get_data_stl(s, i, :split; concat=concat,
            stratified=false, split=vcat(split, 0))
        train = vcat(train, _train)
        valid = vcat(valid, _valid)
    end

    if concat
        train, valid = _concatDicts(train), _concatDicts(valid)
        if simplify
            rmv(x) = (length(x) == 1) ? x[1] : x
            train, valid, test = rmv(train), rmv(valid), rmv(test)
        end
    end

    if splittype == :split
        return train, valid, test
    elseif splittype == :train
        return train
    elseif splittype == :valid
        return valid
    else
        error("Unreachable error")
    end

end



"""
    create_data_split(Y, U, Yraw, split=[0.7,0.1,0.2])

Create a partition of the data into train/validation/test components. The
default size of these partitions is 70%/10%/20% respectively. This can be
switched up by supplying the argument `split=[a, b, c]` s.t. sum(a,b,c) = 1.

It is assumed that *columns* of Y and U contain datapoints, and that *rows* of
Yraw contain datapoints. The output will be a Dict containing the following fields:

* **:Y**    - the observation matrix (each column is an observation).
* **:U**    - the input matrix (each column is a datapoint).
* **:Yrawv - the raw data before standardisation and another manipulation. (Possibly legacy?)
"""
function create_data_split(Y::Matrix{T}, U::Matrix{T}, Yraw::Matrix{T}, split=[0.7,0.1,0.2]) where T
    N = size(Y,2)
    @argcheck size(Y, 2) == size(U, 2)
    @argcheck sum(split) == 1
    int_split = zeros(Int, 3)
    int_split[1] = Int(round(N*split[1]))
    int_split[2] = Int(round(N*split[2]))
    int_split[3] = N - sum(int_split[1:2])
    split = vcat(0, cumsum(int_split), N) .+1
    train, valid, test = (Dict(:Y=>Y[:,split[i]:split[i+1]-1],
                               :U=>U[:,split[i]:split[i+1]-1],
                               :Yraw=>Yraw[split[i]:split[i+1]-1,:]) for i in 1:3)
    return train, valid, test
end

function create_data_split(Y::Vector{MT}, U::Vector{MT}, Yraw::Vector{MT},
    split=[0.7,0.1,0.2]) where MT <: AbstractMatrix
    @argcheck length(Y) == length(U) == length(Yraw)
    train, valid, test = Dict[], Dict[], Dict[]
    for i = 1:length(Y)
        _train, _valid, _test = create_data_split(Y[i], U[i], Yraw[i], split);
        push!(train, _train)
        push!(valid, _valid)
        push!(test, _test)
    end
    return train, valid, test
end


#= -----------------------------------------------------------------------
                Utilities for utilities for accessing data  :S
    This whole thing got way out of hand. I'm sure this can be tidied up =>
    is a function of setting out without a plan, under the assumption that
    "it wouldn't take long". You know what I'm talking about.
   ----------------------------------------------------------------------- =#

function _concat(Ysraw::Vector{MT}, Ys::Vector{MT}, Us::Vector{MT}, simplify::Bool) where MT <: AbstractMatrix
    v = simplify ? identity : x -> [x]
    return v(reduce(vcat, Ysraw)), v(reduce(hcat, Ys)), v(reduce(hcat, Us))
end

function _concat(trainsplit::Vector{D}, simplify::Bool) where D <: Dict
    v = simplify ? identity : x -> [x]
    Ysraw = reduce(vcat, [x[:Yraw] for x in trainsplit])
    Ys = reduce(hcat, [x[:Y] for x in trainsplit])
    Us = reduce(hcat, [x[:U] for x in trainsplit])
    return v(Ysraw), v(Ys), v(Us)
end

function _concatDicts(trainsplit::Vector{D}) where D <: Dict
    Ysraw, Ys, Us = _concat(trainsplit, true)
    return Dict(:Y=>Ys, :U=>Us, :Yraw=>Ysraw)
end

function _concatDicts(train::Vector{D}, valid::Vector{D}, test::Vector{D}) where D <: Dict
    train = _concatDicts(train)
    valid = _concatDicts(valid)
    test  = _concatDicts(test)
    return train, valid, test
end

function _unconcat(Yraw::AbstractMatrix{T}, Y::AbstractMatrix{T}, U::AbstractMatrix{T},
    breaks::Vector{I}) where {T <: Real, I <: Int}

    N = size(Y,2)
    @argcheck size(Y, 2) == size(U, 2) == size(Yraw, 1) == sum(breaks)
    nsplit = length(breaks)
    split = cumsum([0; breaks]) .+ 1
    Ys = [Y[:,split[i]:split[i+1]-1] for i in 1:nsplit]
    Us = [U[:,split[i]:split[i+1]-1] for i in 1:nsplit]
    Ysraw = [Yraw[split[i]:split[i+1]-1, :] for i in 1:nsplit]
    return Ysraw, Ys, Us
end

function _unconcatDict(d::Dict, breaks::Vector{I}) where I <: Int
    Ysraw, Ys, Us = _unconcat(d[:Yraw], d[:Y], d[:U], breaks)
    return [Dict(:Y=>y, :U=>u, :Yraw=>yraw) for (y,u,yraw) in zip(Ys, Us, Ysraw)]
end

# TODO: DRY fail here.

function _unconcatDicts(ds::Vector{D}, breaks::Vector{I}) where {D <: Dict, I <: Int}
    N = length(ds)
    Ls = [size(d[:Yraw], 1) for d in ds]
    @assert sum(Ls) == sum(breaks)
    bs  = copy(breaks)
    cbs = cumsum(breaks)
    out = Dict[]
    for nn in 1:N
        ix = something(findlast(cbs .<= Ls[nn]), 0)
        (ix > 0 && ix < length(bs) && cbs[ix] != Ls[nn]) && (ix += 1)
        ix = max(ix, 1)
        cbreaks = bs[1:ix]
        cbreaks[end] = Ls[nn] - (ix > 1 ? cbs[ix-1] : 0)
        bs = bs[ix:end]
        bs[1] -= cbreaks[end]
        cbs = cbs[ix:end] .- Ls[nn]

        out = vcat(out, _unconcatDict(ds[nn], cbreaks))
    end
    return out
end

function _unconcatDicts(train::Vector{D}, valid::Vector{D}, test::Vector{D},
    breaks::Vector{I}) where {D <: Dict, I <: Int}
    Ls = [sum([size(d[:Y], 2) for d in data]) for data in [train, valid, test]]
    bs  = copy(breaks)
    cbs = cumsum(breaks)
    out = []
    for (nn, data) in enumerate([train, valid, test])
        ix = something(findlast(cbs .<= Ls[nn]), 0)
        (ix > 0 && ix < length(bs) && cbs[ix] != Ls[nn]) && (ix += 1)
        ix = max(ix, 1)
        cbreaks = bs[1:ix]
        cbreaks[end] = Ls[nn] - (ix > 1 ? cbs[ix-1] : 0)
        bs = bs[ix:end]
        bs[1] -= cbreaks[end]
        cbs = cbs[ix:end] .- Ls[nn]
        push!(out, _unconcatDicts(data, cbreaks))
    end
    return out[1], out[2], out[3]
end


function _create_y_u_raw_dict(Ys::Vector{MT}, Us::Vector{MT}, Ysraw::Vector{MT}
    ) where MT <: AbstractMatrix
    [Dict(:Y=>y, :U=>u, :Yraw=>yraw) for (y,u,yraw) in zip(Ys, Us, Ysraw)]
end

function _create_y_u_raw_dict(Ys::AbstractMatrix, Us::AbstractMatrix, Ysraw::AbstractMatrix)
    [Dict(:Y=>Ys, :U=>Us, :Yraw=>Ysraw)]
end



function restrict_expmtdata(expmtdata, N)
    # restrict training set size to N chunks of 64 per style.
    Ys, test_Ys = [], []
    Ysraw, test_Ysraw = [], []
    Us, test_Us = [], []
    num_styles = length(expmtdata.ix_lkp)
    lkp, test_lkp = [[] for i in 1:num_styles], [[] for i in 1:num_styles]

    for s in 1:num_styles
        Nrem = N
        for i in expmtdata.ix_lkp[s]
            y, yraw, u = expmtdata.Ys[i], expmtdata.YsRaw[i], expmtdata.Us[i]
            n = size(y, 2) ÷ 64
            ncur = min(n, Nrem)
            if ncur == 0
                push!(test_Ys, y)
                push!(test_Ysraw, yraw)
                push!(test_Us, u)
                push!(test_lkp[s], i)
            elseif ncur == n
                push!(Ys, y)
                push!(Ysraw, yraw)
                push!(Us, u)
                push!(lkp[s], i)
            else
                push!(Ys, y[:, 1:64*ncur])
                push!(Ysraw, yraw[1:64*ncur, :])
                push!(Us, u[:, 1:64*ncur])
                push!(test_Ys, y[:, 64*ncur+1:end])
                push!(test_Ysraw, yraw[64*ncur+1:end, :])
                push!(test_Us, u[:, 64*ncur+1:end])
                push!(test_lkp[s], i)
                push!(lkp[s], i)
            end
            Nrem -= ncur
        end
    end

    return (ExperimentData(Ysraw, Ys, Us, lkp),
        ExperimentData(test_Ysraw, test_Ys, test_Us, test_lkp))
end
end