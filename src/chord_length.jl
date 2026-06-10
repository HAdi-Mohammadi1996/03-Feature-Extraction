include("utils.jl")
function random_direction(rng)
    z = 2rand(rng) - 1
    theta = 2pi * rand(rng)
    r = sqrt(max(0.0, 1 - z^2))
    return (r * cos(theta), r * sin(theta), z)
end

function ray_box_interval(p, d, dims)
    tmin = -Inf
    tmax = Inf
    for a in 1:3
        if abs(d[a]) < eps(Float64)
            (0 <= p[a] <= dims[a]) || return nothing
        else
            t1 = (0 - p[a]) / d[a]
            t2 = (dims[a] - p[a]) / d[a]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        end
    end
    return tmax > tmin ? (tmin, tmax) : nothing
end

function traverse_ray_chords!(sums, counts, C, p, d, min_chord)
    dims = size(C)
    interval = ray_box_interval(p, d, dims)
    interval === nothing && return
    t0, t1 = interval
    entry = ntuple(a -> p[a] + (t0 + 1e-9) * d[a], 3)
    cell = [clamp(floor(Int, entry[a]) + 1, 1, dims[a]) for a in 1:3]
    step = [d[a] > 0 ? 1 : d[a] < 0 ? -1 : 0 for a in 1:3]
    tdelta = [step[a] == 0 ? Inf : abs(1 / d[a]) for a in 1:3]
    tmax = Vector{Float64}(undef, 3)
    for a in 1:3
        if step[a] > 0
            boundary = cell[a]
            tmax[a] = (boundary - entry[a]) / d[a]
        elseif step[a] < 0
            boundary = cell[a] - 1
            tmax[a] = (boundary - entry[a]) / d[a]
        else
            tmax[a] = Inf
        end
    end

    remaining = t1 - t0
    t = 0.0
    phase = C[cell...]
    chord = 0.0
    while inbounds3((cell[1], cell[2], cell[3]), dims) && t < remaining
        nextt = min(minimum(tmax), remaining)
        segment = max(0.0, nextt - t)
        newphase = C[cell...]
        if newphase == phase
            chord += segment
        else
            if chord >= min_chord && haskey(sums, phase)
                sums[phase] += chord
                counts[phase] += 1
            end
            phase = newphase
            chord = segment
        end
        t = nextt
        for a in 1:3
            if tmax[a] <= nextt + 1e-10
                cell[a] += step[a]
                tmax[a] += tdelta[a]
            end
        end
    end
    if chord >= min_chord && haskey(sums, phase)
        sums[phase] += chord
        counts[phase] += 1
    end
end

function mean_chord_lengths(C::AbstractArray{<:Integer,3}, phases=PHASES;
    Nrays=5000, min_chord=4, voxel_size=0.1, rng=MersenneTwister(1))
    dims = size(C)
    sums = Dict(phase => 0.0 for phase in phases)
    counts = Dict(phase => 0 for phase in phases)
    for _ in 1:Nrays
        p = (rand(rng) * dims[1], rand(rng) * dims[2], rand(rng) * dims[3])
        d = random_direction(rng)
        traverse_ray_chords!(sums, counts, C, p, d, min_chord)
    end
    return Dict(phase => counts[phase] == 0 ? NaN : voxel_size * sums[phase] / counts[phase]
        for phase in phases)
end
