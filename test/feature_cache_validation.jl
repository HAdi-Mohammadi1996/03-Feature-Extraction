using Test

include(joinpath(@__DIR__, "..", "src", "load_feature_extraction.jl"))

function cache_test_geometry()
    C = fill(Int8(1), 12, 8, 6)
    C[5:8, :, :] .= 2
    C[9:12, :, :] .= 3
    return C
end

function validate_feature_cache()
    config = FeatureConfig(
        voxel_size=1.0,
        Nrays=200,
        min_chord=0.0,
        phases=[1, 2, 3],
        use_all_directions=false,
        direction=1,
        physical_threaded=false,
        recalculate_static_phase_each_timestep=false,
        static_phase=2,
        verify_static_phase=true,
    )

    reference = cache_test_geometry()
    changed = copy(reference)
    changed[1:2, :, :] .= 3
    changed[11:12, :, :] .= 1

    _, cache = extract_features_with_phase_cache(
        reference;
        sample="reference",
        config=config,
    )
    cached_row = extract_features(
        changed;
        sample="changed",
        config=config,
        phase_cache=cache,
    )
    full_row = extract_features(changed; sample="changed", config=config)

    @testset "Static phase feature cache" begin
        for name in propertynames(full_row)
            name == :sample && continue
            @test isequal(getproperty(cached_row, name), getproperty(full_row, name))
        end

        invalid = copy(changed)
        invalid[1, 1, 1] = 2
        @test_throws ErrorException extract_features(
            invalid;
            sample="invalid",
            config=config,
            phase_cache=cache,
        )
    end
end

validate_feature_cache()
