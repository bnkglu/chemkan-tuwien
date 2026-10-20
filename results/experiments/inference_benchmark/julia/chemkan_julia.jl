# ChemKAN inference in Julia (OrdinaryDiffEq Tsit5) with the trained PyTorch weights.
#
# Same models, initial states, 100-point output grid and tolerances as benchmark.py; the
# right-hand side is a hand-written, allocation-free port of the PyTorch model:
#   hydrogen  du/dt = [r, Wlin.r + KANcor(u)] * rate_scale,  r = LeanKAN(tanh(AddKAN(u)))
#   biodiesel dz/dt = [LeanKAN(tanh(AddKAN(z))), 0] * rate_scale
# Gaussian RBF psi = exp(-(x-c)^2 / (2h^2)); AddKAN and KANcor without input tanh, LeanKAN
# with it; LeanKAN multiplies its first n_mu inputs' edges and adds the rest.
# Timed region: normalization of the initial state, ODE problem construction, solve, and
# de-normalization into a physical array. Single-threaded (julia -t 1, BLAS 1 thread).
#
#   julia --project=results/experiments/inference_benchmark/julia -t 1 \
#       results/experiments/inference_benchmark/julia/chemkan_julia.jl hydrogen

using NPZ, JSON3, OrdinaryDiffEqTsit5, LinearAlgebra, Statistics, Dates

BLAS.set_num_threads(1)
const HERE = @__DIR__

struct KANModel{T}
    W1::Array{T,3}          # (hidden, in, K)
    W2::Array{T,3}          # (out, hidden, K)
    Wlin::Vector{T}         # (species,) or empty
    Wcor::Array{T,3}        # (1, in, K) or empty
    centers::Vector{T}
    inv2h2::T
    n_mu::Int
    rate_scale::T
    thermo::Bool
    nstate::Int
    nspecies::Int
end

@inline gauss(x, c, inv2h2) = exp(-(x - c)^2 * inv2h2)

# edge sum  sum_j sum_k W[o,j,k] psi(x_j - c_k)  for one output o (no input tanh)
@inline function addkan(W, o, x, centers, inv2h2)
    s = zero(eltype(W))
    @inbounds for j in axes(W, 2), k in eachindex(centers)
        s += W[o, j, k] * gauss(x[j], centers[k], inv2h2)
    end
    s
end

function rhs!(du, u, m::KANModel{T}, hid) where {T}
    @inbounds begin
        for o in axes(m.W1, 1)
            hid[o] = tanh(addkan(m.W1, o, u, m.centers, m.inv2h2))   # LeanKAN input tanh
        end
        dT = zero(T)
        for i in 1:m.nspecies
            prod_ = one(T); add_ = zero(T)
            for j in axes(m.W2, 2)
                e = zero(T)
                for k in eachindex(m.centers)
                    e += m.W2[i, j, k] * gauss(hid[j], m.centers[k], m.inv2h2)
                end
                if j <= m.n_mu
                    prod_ *= e
                else
                    add_ += e
                end
            end
            r = (m.n_mu > 0 ? prod_ : zero(T)) + add_
            du[i] = r * m.rate_scale
            m.thermo && (dT += m.Wlin[i] * r)
        end
        for i in m.nspecies+1:m.nstate
            du[i] = m.thermo ? (dT + addkan(m.Wcor, 1, u, m.centers, m.inv2h2)) * m.rate_scale : zero(T)
        end
    end
    nothing
end

function load_model(d, ::Type{T}) where {T}
    thermo = haskey(d, "Wlin")
    W1 = T.(d["W1"]); W2 = T.(d["W2"])
    nspecies = size(W2, 1)
    KANModel{T}(W1, W2, thermo ? T.(d["Wlin"]) : T[], thermo ? T.(d["Wcor"]) : zeros(T, 0, 0, 0),
                T.(d["centers"]), T(1 / (2 * d["h"]^2)), Int(d["n_mu"]), T(d["rate_scale"]),
                thermo, size(W1, 2), nspecies)
end

# one trajectory (vector state) / all trajectories in one solve (matrix state, columns)
function make_single(m::KANModel{T}) where {T}
    hid = zeros(T, size(m.W1, 1))
    (du, u, p, t) -> rhs!(du, u, m, hid)
end

function make_batched(m::KANModel{T}) where {T}
    hid = zeros(T, size(m.W1, 1))
    function f!(du, u, p, t)
        @inbounds for c in axes(u, 2)
            rhs!(view(du, :, c), view(u, :, c), m, hid)
        end
    end
end

function solve_single(f, u0phys, umin, span, t, rtol, atol, nout, ::Type{T}) where {T}
    z0 = T.((u0phys .- umin) ./ span)
    prob = ODEProblem{true}(f, z0, (T(t[1]), T(t[end])))
    sol = solve(prob, Tsit5(); reltol=rtol, abstol=atol, saveat=T.(t))
    out = Array{Float64}(undef, length(t), nout)
    for (j, z) in enumerate(sol.u), k in 1:nout
        out[j, k] = z[k] * span[k] + umin[k]
    end
    out
end

function solve_batched(f, U0phys, umin, span, t, rtol, atol, nout, ::Type{T}) where {T}
    Z0 = T.((U0phys' .- umin) ./ span)                 # (k, n)
    prob = ODEProblem{true}(f, Z0, (T(t[1]), T(t[end])))
    sol = solve(prob, Tsit5(); reltol=rtol, abstol=atol, saveat=T.(t))
    n = size(U0phys, 1)
    out = Array{Float64}(undef, length(t), n, nout)
    for (j, Z) in enumerate(sol.u), c in 1:n, k in 1:nout
        out[j, c, k] = Z[k, c] * span[k] + umin[k]
    end
    out
end

function timed(f, warmup, reps)
    local out
    for _ in 1:warmup
        out = f()
    end
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); out = f(); push!(ts, (time_ns() - t0) / 1e9)
    end
    ts, out
end

q(x, p) = quantile(x, p)
stats(x) = Dict("median" => median(x), "q25" => q(x, 0.25), "q75" => q(x, 0.75),
                "iqr" => q(x, 0.75) - q(x, 0.25), "min" => minimum(x), "max" => maximum(x), "n" => length(x))

function main(system; warmup=3, reps=25)
    d = npzread(joinpath(HERE, "inputs_$(system).npz"))
    T = d["float32"] == 1 ? Float32 : Float64
    m = load_model(d, T)
    t = Float64.(d["t"]); umin = Float64.(d["u_min"]); span = Float64.(d["u_max"]) .- umin
    U0 = Float64.(d["u0"]); n = size(U0, 1); nout = Int(d["n_out"])
    fs, fb = make_single(m), make_batched(m)
    res = Dict{String,Any}("system" => system, "precision" => string(T), "conditions" => n,
                           "julia" => string(VERSION), "threads" => Threads.nthreads(),
                           "OrdinaryDiffEqTsit5" => string(pkgversion(OrdinaryDiffEqTsit5)),
                           "measured_at" => string(now()), "runs" => Dict{String,Any}())
    preds = Dict{String,Any}()
    tols = d["chemkan_tols"]
    for j in 1:size(tols, 1)
        rtol, atol = tols[j, 1], tols[j, 2]
        lat = Vector{Vector{Float64}}(); single = Array{Float64}(undef, length(t), n, nout)
        for i in 1:n
            ts, out = timed(() -> solve_single(fs, U0[i, :], umin, span, t, rtol, atol, nout, T), warmup, reps)
            push!(lat, ts); single[:, i, :] = out
        end
        seq, _ = timed(() -> [solve_single(fs, U0[i, :], umin, span, t, rtol, atol, nout, T) for i in 1:n], warmup, reps)
        bat, batched = timed(() -> solve_batched(fb, U0, umin, span, t, rtol, atol, nout, T), warmup, reps)
        key = "julia_chemkan_tol$(j - 1)"
        res["runs"][key] = Dict("rtol" => rtol, "atol" => atol, "latency" => stats(reduce(vcat, lat)),
                                "latency_per_condition_median_s" => median.(lat),
                                "throughput_sequential" => stats(seq), "throughput_batched" => stats(bat))
        preds["single_$(j - 1)"] = single; preds["batched_$(j - 1)"] = batched
        println("$key rtol $rtol: latency median $(round(median(reduce(vcat, lat)) * 1e3, digits=3)) ms, " *
                "sequential $(round(median(seq), digits=4)) s, batched $(round(median(bat), digits=4)) s")
    end
    open(joinpath(HERE, "julia_chemkan_$(system).json"), "w") do io
        JSON3.pretty(io, res)
    end
    npzwrite(joinpath(HERE, "julia_chemkan_$(system)_pred.npz"), preds)
end

main(length(ARGS) > 0 ? ARGS[1] : "hydrogen"; reps=length(ARGS) > 1 ? parse(Int, ARGS[2]) : 25)
