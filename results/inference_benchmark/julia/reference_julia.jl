# Detailed-chemistry references in Julia, on the same initial-value problems as benchmark.py.
#
# hydrogen / methane  Arrhenius.jl (GitHub master; mechanism/h2o2.yaml + .npz from make_arrhenius_mech.py,
#           i.e. Cantera's h2o2.yaml with N2 and AR), adiabatic constant-pressure 0-D reactor
#           written as in Arrhenius.jl's ignition example; Sundials CVODE_BDF.
# biodiesel the released example's trueODEfunc and Arrhenius rate constants
#           (ChemKAN_biodiesel_example.jl), Float64; the example's data-generation solver
#           AutoTsit5(Rosenbrock23(autodiff=AutoFiniteDiff())) at its default tolerances, and CVODE_BDF at
#           the Python reference tolerance.
# Timed region per trajectory: initial state, ODEProblem, solve on the 100-point grid, output
# array (kept species + T for hydrogen). julia -t 1, BLAS 1 thread.
#
#   julia --project=results/experiments/inference_benchmark/julia -t 1 \
#       results/experiments/inference_benchmark/julia/reference_julia.jl hydrogen

using Arrhenius, Sundials, OrdinaryDiffEqTsit5, OrdinaryDiffEqRosenbrock, NPZ, JSON3
using LinearAlgebra, Statistics, Dates
using ADTypes: AutoFiniteDiff

BLAS.set_num_threads(1)
const HERE = @__DIR__

q(x, p) = quantile(x, p)
stats(x) = Dict("median" => median(x), "q25" => q(x, 0.25), "q75" => q(x, 0.75),
                "iqr" => q(x, 0.75) - q(x, 0.25), "min" => minimum(x), "max" => maximum(x), "n" => length(x))

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

# ---------------------------------------------------------------- hydrogen (Arrhenius.jl)
const MECH = Dict("hydrogen" => "h2o2.yaml", "methane" => "gri30.yaml")

function hydrogen_setup(d, system="hydrogen")
    gas = CreateSolution(joinpath(HERE, "mechanism", MECH[system]))
    ns = gas.n_species
    P = Float64(d["pressure"])
    function dudt!(du, u, p, t)                 # Arrhenius.jl ignition example, constant p
        T = u[end]
        Y = @view(u[1:ns])
        mean_MW = 1.0 / dot(Y, 1 ./ gas.MW)
        ρ_mass = P / R / T * mean_MW
        X = Y2X(gas, Y, mean_MW)
        C = Y2C(gas, Y, ρ_mass)
        cp_mole, cp_mass = get_cp(gas, T, X, mean_MW)
        h_mole = get_H(gas, T, Y, X)
        S0 = get_S(gas, T, P, X)
        wdot = wdot_func(gas.reaction, T, C, S0, h_mole)
        Ydot = wdot / ρ_mass .* gas.MW
        Tdot = -dot(h_mole, wdot) / ρ_mass / cp_mass
        du .= vcat(Ydot, Tdot)
    end
    keep = Int.(d["keep"]) .+ 1                     # Cantera indices of the 9 kept species
    names = gas.species_names
    (dudt!, keep, ns, names)
end

function hydrogen_solve(setup, y0, T0, t, alg, rtol, atol)
    dudt!, keep, ns, _ = setup
    u0 = vcat(y0, T0)
    prob = ODEProblem(dudt!, u0, (t[1], t[end]))
    sol = solve(prob, alg; reltol=rtol, abstol=atol, saveat=t)
    sol.retcode == ReturnCode.Success || error("retcode $(sol.retcode)")
    out = Array{Float64}(undef, length(t), length(keep) + 1)
    for (j, u) in enumerate(sol.u)
        out[j, 1:end-1] .= u[keep]; out[j, end] = u[end]
    end
    out
end

# ---------------------------------------------------------------- biodiesel (released example)
function trueODEfunc(dydt, y, k, t)
    r1 = k[1] * y[1] * y[2]
    r2 = k[2] * y[3] * y[2]
    r3 = k[3] * y[4] * y[2]
    dydt[1] = -r1
    dydt[2] = -r1 - r2 - r3
    dydt[3] = r1 - r2
    dydt[4] = r2 - r3
    dydt[5] = r3
    dydt[6] = r1 + r2 + r3
    dydt[7] = 0.0
end
const logA = [18.60, 19.13, 7.93]
const Ea = [14.54, 14.42, 6.47]                     # kcal/mol
rates(T) = exp.(logA) .* exp.(-Ea ./ 1.98720425864083e-3 ./ T)

function biodiesel_solve(u0, t, alg, rtol, atol)
    prob = ODEProblem(trueODEfunc, copy(u0), (t[1], t[end]), rates(u0[end]))
    sol = rtol === nothing ? solve(prob, alg; saveat=t) :
          solve(prob, alg; reltol=rtol, abstol=atol, saveat=t)
    sol.retcode == ReturnCode.Success || error("retcode $(sol.retcode)")
    out = Array{Float64}(undef, length(t), 6)
    for (j, u) in enumerate(sol.u)
        out[j, :] .= @view u[1:6]
    end
    out
end

function main(system; warmup=3, reps=25)
    d = npzread(joinpath(HERE, "inputs_$(system).npz"))
    t = Float64.(d["t"]); U0 = Float64.(d["u0"]); n = size(U0, 1)
    if system in ("hydrogen", "methane")
        setup = hydrogen_setup(d, system)
        Y0 = Float64.(d["Y0_full"]); T0 = Float64.(d["T0"])
        settings = [("cvode_bdf_rtol1e-9_atol1e-15", CVODE_BDF(), 1e-9, 1e-15),
                    ("cvode_bdf_rtol1e-6_atol1e-12", CVODE_BDF(), 1e-6, 1e-12),
                    ("cvode_bdf_rtol1e-6_atol1e-10", CVODE_BDF(), 1e-6, 1e-10),
                    ("cvode_bdf_rtol1e-6_atol1e-8", CVODE_BDF(), 1e-6, 1e-8),
                    ("cvode_bdf_rtol1e-2_atol1e-6", CVODE_BDF(), 1e-2, 1e-6)]
        if system == "methane"          # GRI-Mech 3.0: the generation tolerance, a practical one, the loose one
            settings = settings[[1, 2, 5]]
        end
        solve1 = (i, alg, rtol, atol) -> hydrogen_solve(setup, Y0[i, :], T0[i], t, alg, rtol, atol)
        nout = length(setup[2]) + 1
        label = "Arrhenius.jl $(pkgversion(Arrhenius)) (GitHub master) + Sundials CVODE_BDF"
    else
        settings = [("cvode_bdf_rtol1e-10_atol1e-12", CVODE_BDF(), 1e-10, 1e-12),
                    ("autotsit5_rosenbrock23_default_tol", AutoTsit5(Rosenbrock23(autodiff=AutoFiniteDiff())), nothing, nothing)]
        solve1 = (i, alg, rtol, atol) -> biodiesel_solve(U0[i, :], t, alg, rtol, atol)
        nout = 6
        label = "released-example kinetics (trueODEfunc), Float64"
    end
    res = Dict{String,Any}("system" => system, "reference" => label, "conditions" => n,
                           "julia" => string(VERSION), "threads" => Threads.nthreads(),
                           "Sundials" => string(pkgversion(Sundials)),
                           "measured_at" => string(now()), "runs" => Dict{String,Any}())
    preds = Dict{String,Any}()
    for (key, alg, rtol, atol) in settings
        failed = Int[]
        for i in 1:n
            try
                solve1(i, alg, rtol, atol)
            catch
                push!(failed, i - 1)
            end
        end
        if !isempty(failed)
            res["runs"][key] = Dict("rtol" => rtol, "atol" => atol, "failed_conditions" => failed)
            println("$key: failed for conditions $failed; not timed")
            continue
        end
        lat = Vector{Vector{Float64}}(); pred = Array{Float64}(undef, length(t), n, nout)
        for i in 1:n
            ts, out = timed(() -> solve1(i, alg, rtol, atol), warmup, reps)
            push!(lat, ts); pred[:, i, :] = out
        end
        seq, _ = timed(() -> [solve1(i, alg, rtol, atol) for i in 1:n], warmup, reps)
        res["runs"][key] = Dict("rtol" => rtol, "atol" => atol, "latency" => stats(reduce(vcat, lat)),
                                "latency_per_condition_median_s" => median.(lat),
                                "throughput_sequential" => stats(seq))
        preds[key] = pred
        println("$key: latency median $(round(median(reduce(vcat, lat)) * 1e3, digits=3)) ms, " *
                "sequential $(round(median(seq), digits=4)) s")
    end
    open(joinpath(HERE, "julia_reference_$(system).json"), "w") do io
        JSON3.pretty(io, res)
    end
    npzwrite(joinpath(HERE, "julia_reference_$(system)_pred.npz"), preds)
end

main(length(ARGS) > 0 ? ARGS[1] : "hydrogen"; reps=length(ARGS) > 1 ? parse(Int, ARGS[2]) : 25)
