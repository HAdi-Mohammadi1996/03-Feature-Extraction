function removing_isolated_particles(C, phase, direction)
    direction ∈ 1:ndims(C) || error("Invalid transport direction")

    mask = C .== phase
    labels = label_components(mask)

    inlet_labels = unique(selectdim(labels, direction, 1))
    outlet_labels = unique(selectdim(labels, direction, size(C, direction)))

    boundary_labels = union(inlet_labels, outlet_labels)
    boundary_labels = boundary_labels[boundary_labels .!= 0]
    boundary_set = BitSet(boundary_labels)
    connected_labels = in.(labels, Ref(boundary_set))
    
    return mask .& connected_labels
end

function is_percolated(C, phase, direction)
    direction ∈ 1:ndims(C) || error("Invalid transport direction")

    mask = C .== phase
    labels = label_components(mask)

    inlet_labels = unique(selectdim(labels, direction, 1))
    outlet_labels = unique(selectdim(labels, direction, size(C, direction)))

    percolating_labels = intersect(inlet_labels, outlet_labels)
    percolating_labels = percolating_labels[percolating_labels .!= 0]

    return !isempty(percolating_labels)
end
