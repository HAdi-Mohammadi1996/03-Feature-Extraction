const FEATURE_EXTRACTION_CHILD = "FEATURE_EXTRACTION_THREADED_CHILD"

"""
    ensure_compute_threads(script_path, requested_threads)

Relaunch `script_path` with the requested number of Julia compute threads when
the current Run-button process has fewer threads. Returns `true` in the process
that should perform the calculation and `false` in the parent process after the
threaded child has finished.
"""
function ensure_compute_threads(script_path::AbstractString, requested_threads::Integer)
    requested_threads > 0 || error("requested_threads must be positive")

    current_threads = Threads.nthreads(:default)
    current_threads >= requested_threads && return true

    if get(ENV, FEATURE_EXTRACTION_CHILD, "0") == "1"
        @warn "Requested $requested_threads compute threads, but only $current_threads are available."
        return true
    end

    script = abspath(script_path)
    isfile(script) || error("Cannot relaunch missing script: $script")

    println(
        "Restarting automatically with $requested_threads Julia compute threads ",
        "(currently $current_threads)...",
    )
    command = `$(Base.julia_cmd()) --threads=$(requested_threads),1 $script`
    run(addenv(command, FEATURE_EXTRACTION_CHILD => "1"))
    return false
end
