const PHASES = (1, 2, 3)
const FACE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
const NEIGHBOR26 = [(dx, dy, dz) for dx in -1:1 for dy in -1:1 for dz in -1:1
    if !(dx == 0 && dy == 0 && dz == 0)]

struct PercolationResult
    fraction::Float64
    mask::BitArray{3}
end

inbounds3(I, dims) = 1 <= I[1] <= dims[1] && 1 <= I[2] <= dims[2] && 1 <= I[3] <= dims[3]
lin(I, dims) = LinearIndices(dims)[I...]
coord(i, dims) = Tuple(CartesianIndices(dims)[i])

function axis_value(I, dir)
    return dir == 1 ? I[1] : dir == 2 ? I[2] : I[3]
end

function feature_row(sample, phase, vf, chord, area, gt, pt, perc, tpb, active_tpb)
    return (sample=sample, phase=phase, volume_fraction=vf,
        mean_chord_length_um=chord, surface_area_um2=area,
        geometric_tortuosity=gt, physical_tortuosity=pt,
        percolation_fraction=perc, total_tpb_density_um_inv2=tpb,
        active_tpb_density_um_inv2=active_tpb)
end

function mean_skip_inf(values)
    kept = [v for v in values if isfinite(v)]
    return isempty(kept) ? Inf : mean(kept)
end

csv_value(::Missing) = ""
csv_value(x::AbstractString) = x
csv_value(x::Real) = isfinite(x) ? string(x) : "Inf"

function minheap_push!(heap, item)
    push!(heap, item)
    i = length(heap)
    while i > 1
        p = i >>> 1
        heap[p][1] <= heap[i][1] && break
        heap[p], heap[i] = heap[i], heap[p]
        i = p
    end
    return heap
end

function minheap_pop!(heap)
    item = heap[1]
    last = pop!(heap)
    if !isempty(heap)
        heap[1] = last
        i = 1
        while true
            l = i << 1
            r = l + 1
            l > length(heap) && break
            c = (r <= length(heap) && heap[r][1] < heap[l][1]) ? r : l
            heap[i][1] <= heap[c][1] && break
            heap[i], heap[c] = heap[c], heap[i]
            i = c
        end
    end
    return item
end
