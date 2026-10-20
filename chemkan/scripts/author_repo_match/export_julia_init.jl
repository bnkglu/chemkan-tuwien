# Export the INITIAL parameters of the authors' released biodiesel example (DENG-MIT/ChemKAN).
#
#   julia +1.11.1 --startup-file=no --project=<authors_dir> export_julia_init.jl <authors_dir> <out_file>
#
# Runs ChemKAN_biodiesel_example_compat.jl verbatim up to and including
#   p = Float32.((deepcopy(pM_data)))
# i.e. Random.seed!(1234), the data generation (whose randn call advances the RNG even at
# zero noise) and Lux.setup(rng, kan1), so `p` is exactly the vector the training loop
# starts from. Nothing before that line writes files; nothing in <authors_dir> is written.
# <out_file>: 156 Float32 values, ComponentArray data order (same layout as p.txt).

using DelimitedFiles

authors, out = abspath(ARGS[1]), abspath(ARGS[2])
script = joinpath(authors, "ChemKAN_biodiesel_example_compat.jl")
lines = split(read(script, String), '\n')
stop = findfirst(l -> strip(l) == "p = Float32.((deepcopy(pM_data)))", lines)
stop === nothing && error("initialization line not found in $script")

cd(authors)                                  # readdlm uses relative paths
task_local_storage(:SOURCE_PATH, script) do  # so include("src/...") resolves in <authors_dir>
    include_string(Main, join(lines[1:stop], '\n'), script)
end

@assert length(Main.p) == 156 && eltype(Main.p) == Float32
mkpath(dirname(out))
writedlm(out, Main.p)
println("wrote ", length(Main.p), " initial parameters (script lines 1-", stop, ") to ", out)
