using ImageFiltering

function specific_surface_area(C, phase; spacing=ntuple(_ -> 1.0, ndims(C)), σ=1.0)
    mask = Float64.(C .== phase)
    smoothed_mask = σ==0.0 ? mask : imfilter(mask, KernelFactors.gaussian(ntuple(_ -> σ, ndims(C))), "replicate")
    G = imgradients(smoothed_mask, KernelFactors.scharr, "replicate")
    normG = sqrt.(sum((G[d] ./ spacing[d]).^2 for d in 1:ndims(C)))
    return sum(normG) / length(C)
end
