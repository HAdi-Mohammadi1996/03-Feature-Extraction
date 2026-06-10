function volume_fractions(C::AbstractArray{<:Integer,3}, phases=PHASES)
    n = length(C)
    return Dict(phase => count(==(phase), C) / n for phase in phases)
end
