# Export the authors' released biodiesel example (DENG-MIT/ChemKAN) for the PyTorch
# equivalence check and the authors-repo-matched PyTorch run.
#
#   julia +1.11.1 --startup-file=no --project=<authors_dir> export_julia_reference.jl <authors_dir> <out_dir>
#
# Rebuilds the example's data, normalization and model exactly as
# ChemKAN_biodiesel_example_compat.jl does (the approved three-line API compatibility copy
# of the released file), loads the trained checkpoint, and writes plain-text arrays.
# Nothing in <authors_dir> is written.
#
# Array layouts (all written with DelimitedFiles.writedlm, full precision):
#   ode_data.txt, normdata.txt, pred_final_p.txt : 240 x 30, row (e-1)*6 + s = experiment e, species s
#   u0_norm.txt   : 40 x 7 normalized initial states [Y1..Y6, T]
#   p.txt         : 156 flat parameters (ComponentArray data order)
#   states.txt    : K x 7 probe states;  layer1.txt K x 4;  kan_out.txt K x 6;  rhs.txt K x 7

using DiffEqFlux, OrdinaryDiffEq, Flux
using SciMLSensitivity
using ComponentArrays
using DelimitedFiles
using Random
using BSON
const Optimisers = Flux.Optimisers          # BSON needs this binding to rebuild `opt`

authors, out = ARGS[1], ARGS[2]
mkpath(out)

rng = Random.default_rng()
Random.seed!(1234);

# ---- settings copied from the example ----------------------------------------------
datasize = 30; tstep = 1; n_exp_train = 20; n_exp_test = 10
n_exp = n_exp_train + n_exp_test; noise = 0.00; ns = 6
alg = AutoTsit5(Rosenbrock23(autodiff=AutoFiniteDiff()));
atol = 1e-5; rtol = 1e-2

function trueODEfunc(dydt, y, k, t)
    r1 = k[1] * y[1] * y[2]; r2 = k[2] * y[3] * y[2]; r3 = k[3] * y[4] * y[2]
    dydt[1] = -r1; dydt[2] = -r1 - r2 - r3; dydt[3] = r1 - r2
    dydt[4] = r2 - r3; dydt[5] = r3; dydt[6] = r1 + r2 + r3; dydt[7] = 0.f0
end
logA = Float32[18.60f0, 19.13f0, 7.93f0]
Ea = Float32[14.54f0, 14.42f0, 6.47f0]
Arrhenius(logA, Ea, T) = exp.(logA) .* exp.(-Ea ./ 1.98720425864083f-3 ./ T)

u0_list = readdlm(joinpath(authors, "diesel_u0_samples.txt"))
u0_list[:, 1:2] = u0_list[:, 1:2] .* 1.5 .+ 0.5
u0_list[:, 3:ns] .= 0.0
u0_list[:, ns + 1] = u0_list[:, ns + 1] .* 20.0 .+ 323.0
u0_list = [u0_list; u0_list[n_exp_train+1:n_exp_train+n_exp_test, :]]
n_exp = n_exp + n_exp_test
u0_raw = copy(u0_list)

tspan = Float32[0.0, datasize * tstep]
tsteps = range(tspan[1], tspan[2], length=datasize)
ode_data_list = zeros(Float32, (n_exp, ns, datasize))
for i in 1:n_exp
    u0 = u0_list[i, :]
    k = Arrhenius(logA, Ea, u0[end])
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k)
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps))[1:end - 1, :]
    maxes = maximum(ode_data, dims=2)
    if i <= n_exp_test + n_exp_train
        ode_data += randn(size(ode_data)) .* maxes .* noise
    end
    ode_data = max.(ode_data, 0)
    ode_data_list[i, :, :] = ode_data
end

include(joinpath(authors, "src/KolmogorovArnold_mult_add.jl"))
using .KolmogorovArnold_mult_add

layer_width = 4; grid_size_1 = 3; grid_size_2 = 3
basis_func = rbf; normalizer = tanh_fast
kan1 = Lux.Chain(
    KDense(7, layer_width, grid_size_1; use_base_act=false, basis_func, normalizer=false, mult_flag=false),
    KDense(layer_width, 6, grid_size_2; use_base_act=false, basis_func, normalizer, mult_flag=true),
)
pM, stM = Lux.setup(rng, kan1)
pM_axis = getaxes(ComponentArray(pM))

ymax = maximum(maximum(ode_data_list, dims=1), dims=3)
ymin = minimum(minimum(ode_data_list, dims=1), dims=3)
normdata = (ode_data_list .- ymin) ./ (ymax - ymin)
all_temps = u0_list[:, 7]
Tmin, Tmax = minimum(all_temps), maximum(all_temps)
u0_list[:, 7] = (u0_list[:, 7] .- Tmin) ./ (Tmax - Tmin)
u0_list[:, 1:6] = (u0_list[:, 1:6] .- ymin[1, :, 1]') ./ (ymax[1, :, 1]' - ymin[1, :, 1]')

dudt!(du, u, p, t) = (du .= vcat(kan1(u, ComponentArray(p, pM_axis), stM)[1], 0.f0) ./ 50)
prob = ODEProblem(dudt!, u0_list[1, :], tspan, saveat=tsteps, abstol=atol, reltol=rtol)
predict_n_ode(u0, p) = Array(solve(remake(prob, u0=u0, p=p), alg, abstol=1e-6))
mse(a, b) = sum(abs2, a .- b) / length(a)
loss_n_ode(p, i) = mse(predict_n_ode(u0_list[i, :], p)[1:6, :], normdata[i, 1:6, :])
agg(p, r) = sum(loss_n_ode(p, i) for i in r) / length(r)

# ---- trained checkpoint -------------------------------------------------------------
ck = BSON.load(joinpath(authors, "checkpoint/mymodel_0_noise.bson"), @__MODULE__)
p = ck[:p]
@assert length(p) == 156
@assert p == ck[:p_opt]
ps = ComponentArray(p, pM_axis)

# ---- probe states: normalized test states at several times, plus off-grid states ----
probe = Vector{Vector{Float64}}()
for i in n_exp_train+1:n_exp_train+n_exp_test, j in (1, 5, 10, 20, 30)
    push!(probe, vcat(Float64.(normdata[i, :, j]), u0_list[i, 7]))
end
prng = MersenneTwister(7)
for _ in 1:10
    push!(probe, -0.2 .+ 1.4 .* rand(prng, 7))            # also exercise the RBF tails
end
S = reduce(hcat, probe)'                                  # K x 7
K = size(S, 1)
L1 = zeros(K, 4); KO = zeros(K, 6); RHS = zeros(K, 7)
for k in 1:K
    z = collect(S[k, :])
    L1[k, :] = kan1[1](z, ps.layer_1, stM.layer_1)[1]
    KO[k, :] = kan1(z, ps, stM)[1]
    du = similar(z); dudt!(du, z, p, 0.0); RHS[k, :] = du
end

flat(A) = reduce(vcat, [A[e, :, :] for e in 1:size(A, 1)])   # (E,S,T) -> (E*S, T), row (e-1)*S+s
pred = zeros(n_exp, 6, datasize)
for i in 1:n_exp
    pred[i, :, :] = predict_n_ode(u0_list[i, :], p)[1:6, :]
end

writedlm(joinpath(out, "p.txt"), p)
writedlm(joinpath(out, "u0_raw.txt"), u0_raw)
writedlm(joinpath(out, "u0_norm.txt"), u0_list)
writedlm(joinpath(out, "ode_data.txt"), flat(ode_data_list))
writedlm(joinpath(out, "normdata.txt"), flat(normdata))
writedlm(joinpath(out, "ymin.txt"), vec(ymin)); writedlm(joinpath(out, "ymax.txt"), vec(ymax))
writedlm(joinpath(out, "T_minmax.txt"), [Tmin, Tmax])
writedlm(joinpath(out, "tsteps.txt"), collect(tsteps))
writedlm(joinpath(out, "states.txt"), S); writedlm(joinpath(out, "layer1.txt"), L1)
writedlm(joinpath(out, "kan_out.txt"), KO); writedlm(joinpath(out, "rhs.txt"), RHS)
writedlm(joinpath(out, "pred_final_p.txt"), flat(pred))
for key in (:list_loss_train, :list_loss_val, :list_loss_val_noisefree)
    writedlm(joinpath(out, "$(key).txt"), Float64.(ck[key]))
end
open(joinpath(out, "julia_meta.txt"), "w") do f
    println(f, "julia_version=", VERSION)
    println(f, "param_axis=", pM_axis)
    println(f, "checkpoint_iter=", ck[:iter])
    println(f, "loss_train_at_final_p=", agg(p, 1:20))
    println(f, "loss_val_at_final_p=", agg(p, 21:30))
    println(f, "loss_val_noisefree_at_final_p=", agg(p, 31:40))
    println(f, "loss_train_last_logged=", ck[:list_loss_train][end])
    println(f, "loss_val_last_logged=", ck[:list_loss_val][end])
end
println("exported ", K, " probe states to ", out)
