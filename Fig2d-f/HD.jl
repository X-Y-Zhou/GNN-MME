# Create the Hellinger Dist table of Neural-ODE
using Flux, DifferentialEquations
using Distances
using DelimitedFiles, LinearAlgebra, StatsBase, Statistics
using BSON: @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
include("aux_functions.jl")

# Truncation
N = 39; 

path = "Fig2d-f"
scale_list = [2, 2.5, 3, 3.5, 4, 4.5]
sample_size_list = @. Int(round(10 ^ scale_list))

# Model initialization
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
p = p_inter

# Define Hill function
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

# Generate Di matrix
function sparse_D(ρ, d)  
    M = length(ρ)
    D = Array{Any, 1}(undef, M)
    for m = 1:M
        D[m] =
            spdiagm(-1 => fill(ρ[m], N)) +
            spdiagm(0 => [fill(-ρ[m], N); 0.0]) +
            spdiagm(0 => [-collect(0.0:N-1) .* d[m]; 0.0]) +
            spdiagm(1 => [collect(1.0:N-1) .* d[m]; 0.0])
    end
    D
end

# Define the CME
function CME!(du, u, p, t; Graph, D)
    rep_inter = re_inter(p)
    # Define matrix Ch
    B = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))
    for m = 1:length(D)
        P = @view u[:, m]
        du[:, m] = (D[m] + length(Graph[m]) * B) * P 
        for j in Graph[m]
            P_neighbor = @view u[:, j]
            NN_in = rep_inter([P; P_neighbor])
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)
            du[:, m] += G_in * P
        end
    end
end

# 3 independent repeated experiments
p_path_list = [
    ["p_scale_2.0_v1.bson", "p_scale_2.0_v2.bson", "p_scale_2.0_v3.bson"],
    ["p_scale_2.5_v1.bson", "p_scale_2.5_v2.bson", "p_scale_2.5_v3.bson"],
    ["p_scale_3.0_v1.bson", "p_scale_3.0_v2.bson", "p_scale_3.0_v3.bson"],
    ["p_scale_3.5_v1.bson", "p_scale_3.5_v2.bson", "p_scale_3.5_v3.bson"],
    ["p_scale_4.0_v1.bson", "p_scale_4.0_v2.bson", "p_scale_4.0_v3.bson"],
    ["p_scale_4.5_v1.bson", "p_scale_4.5_v2.bson", "p_scale_4.5_v3.bson"],
]

# Preset parameters of ODE solver
VT = 10;
tf = 20.0;
tspan = (0, tf);
tstep = 0.1; #

# Version 1 ρ=2.5, d=0.5
version = 1
table_list = [[0. for _ in 1:3] for _ in 1:6]

# Read SSA data
SS_proba_list_v1 = Array{Any, 2}(undef, 6, 3)
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]

rho = [2.5 for _ in 1:VT]
d = [0.5 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)

for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        # Load training parameters
        @load "$path/model_params/$(p_path)" p

        # Initialize the ODE solver
        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v1[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), ssa_SS_P)
    end
end
table_v1 = hcat(table_list...)

# version 3 ρ=1.0, d=1.0
version = 3
table_list = [[0. for _ in 1:3] for _ in 1:6]

SS_proba_list_v2 = Array{Any, 2}(undef, 6, 3)
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]

rho = [1.0 for _ in 1:VT]
d = [1.0 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)

for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        @load "$path/model_params/$(p_path)" p

        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v2[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), abs.(ssa_SS_P))
    end
end
table_v2 = hcat(table_list...)

# Plot HD
fig_colors = reshape(palette(:tab10)[:], 1, :);

# Plot version1
version = 1
table_v1_mean = mean(table_v1, dims=1)
table_v1_std = std(table_v1, dims=1)

SSA_dist_10 = readdlm("$path/data/10_cells_SSA_distance_table_v$(version).csv", ',')
SSA_dist_mean_10 = mean(SSA_dist_10, dims=1)
SSA_dist_var_10 = var(SSA_dist_10, dims=1)
SSA_dist_std_10 = std(SSA_dist_10, dims=1)
f1 = plot(title="$VT cells version $version", xscale=:log10, xlims=(10^1.5, 10^5), ylims=(0, 0.25));
f1 = plot!(sample_size_list[1:6], SSA_dist_mean_10[1:6], yerr=SSA_dist_std_10[1:6], label="$(VT) Cells SSA",  msw=2, c=fig_colors[1], msc=fig_colors[1])
f1 = plot!(sample_size_list[1:6], table_v1_mean[1:6], yerr=table_v1_std[1:6], label="$(VT) Cells NeuralODE",  msw=2, c=fig_colors[2], msc=fig_colors[2], st=:scatter, ms=3)


# Plot version3
version = 3
table_v2_mean = mean(table_v2, dims=1)
table_v2_std = std(table_v2, dims=1)

SSA_dist_10 = readdlm("$path/data/10_cells_SSA_distance_table_v$(version).csv", ',')
SSA_dist_mean_10 = mean(SSA_dist_10, dims=1)
SSA_dist_var_10 = var(SSA_dist_10, dims=1)
SSA_dist_std_10 = std(SSA_dist_10, dims=1)
f2 = plot(title="$VT cells version $version", xscale=:log10, xlims=(10^1.5, 10^5), ylims=(0, 0.1));
f2 = plot!(sample_size_list[1:6], SSA_dist_mean_10[1:6], yerr=SSA_dist_std_10[1:6], label="$(VT) Cells SSA",  msw=2, c=fig_colors[1], msc=fig_colors[1])
f2 = plot!(sample_size_list[1:6], table_v2_mean[1:6], yerr=table_v2_std[1:6], label="$(VT) Cells NeuralODE",  msw=2, c=fig_colors[2], msc=fig_colors[2], st=:scatter, ms=3)

plot(f1, f2, size=(800, 300), ylims=(0, 0.25))
# savefig("Results/Fig2d.pdf")

# Plot Nueral-ODE and SSA distribution
#version 1
Figures = Any[]
version = 1 
SSA_proba = [readdlm("$path/data/10_cells_v$(version)/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter,label="GNN-MME")
    f = plot!(SS_proba_list_v1[i, 2],label="SSA")
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 30), ylims=(0, 0.25), size=(1100, 200))
# savefig("Results/Figure2_f_AB.pdf")

#version 3
Figures = Any[]
version = 3  
SSA_proba = [readdlm("$path/data/10_cells_v$(version)/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter,label="GNN-MME")
    f = plot!(SS_proba_list_v2[i, 1],label="SSA")
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 20), ylims=(0, 0.50), size=(1100, 200))
#savefig("Results/Figure2_f_CD.pdf")
