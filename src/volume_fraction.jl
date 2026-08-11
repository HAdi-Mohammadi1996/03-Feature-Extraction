function volume_fraction(C, phase) 
    return count(==(phase), C) / length(C)
end
