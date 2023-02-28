# Code for Whole Cell Model validation
using Flux, DifferentialEquations, Zygote
using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @save, @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
include("./aux_func.jl")

# Truncation
N = 39;

# Load training set
tf = 150
tstep = 0.5
VT = 121

ssa_path = "FigS4/data"
data_path = ["$ssa_path/$(VT)_cells"]
tpoints = readdlm("$(data_path[1])/timepoints.csv", ',')[:]
tstamp = length(tpoints)
VTs = fill(VT, length(data_path))
ssa_proba = Array{Any, 1}(undef, length(data_path))

Threads.@threads for i in 1:length(VTs)
    mRNA_proba_list = [reshape(readdlm("$(data_path[i])/proba/M_$(v).csv", ','), (N+1, 1, tstamp)) for v in 1:VTs[i]]
    Protein_proba_list = [reshape(readdlm("$(data_path[i])/proba/P_$(v).csv", ','), (N+1, 1, tstamp)) for v in 1:VTs[i]]
    ssa_proba[i] = hcat([hcat(mRNA_proba_list[j], Protein_proba_list[j]) for j in 1:VTs[i]]...)
end

# Define Hill function
hill(n::Union{Int64, Float64}; k=10.0, K=5.0) = k / (n + K)
hill(ns::Vector; k=10.0, K=5.0) = [hill(ns[i], k=k, K=K) for i in 1:length(ns)]

# Model initialization and load training parameters
train_path = "FigS4/model_params"

# Define GNNintra1 network
intra1_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                     x -> leakyrelu.(x))
p_intra1, re_intra1 = Flux.destructure(intra1_struc)
@load "$train_path/p_intra1.bson" p_intra1
rep_intra1 = re_intra1(p_intra1)

# Define GNNintra2 network
intra2_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                     x -> leakyrelu.(x))
p_intra2, re_intra2 = Flux.destructure(intra2_struc)
@load "$train_path/p_intra2.bson" p_intra2
rep_intra2 = re_intra2(p_intra2)

# Define mRNA inter network
Minter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                       x -> leakyrelu.(x))
p_Minter, re_Minter = Flux.destructure(Minter_struc)
@load "$train_path/p_Minter.bson" p_Minter
rep_Minter = re_Minter(p_Minter)

# Define Protein inter network
Pinter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                       x -> leakyrelu.(x))
p_Pinter, re_Pinter = Flux.destructure(Pinter_struc)
@load "$train_path/p_Pinter.bson" p_Pinter
rep_Pinter = re_Minter(p_Pinter)

# Define the CME
𝐁 = spdiagm(0 => -collect(0:N)) + spdiagm(1 => collect(1:N))
GNNn(v) = spdiagm(0 => [-v; 0.0]) + spdiagm(-1 => v)

function WholeCell_CME!(du, u, p, t; VT, Cells_sets, graphs)
    # Get the parameters of the cell
    k, K, λ, dn1, dn2, dc1, dc2 = p[1:7]
    Dn1, Dn2, Dc1, Dc2, Dnc1, Dnc2 = p[8:13]
    Gene_Cell = Cells_sets[1]
    Nucleus_Cells, Cytoplasm_Cells = Cells_sets[2:3]
    Nucleus_Boundary_Cells, Cytoplasm_Boundary_Cells = Cells_sets[4:5]
    mRNA_Nucleus_graph, mRNA_Cytoplasm_graph, mRNA_Boundary_graph = graphs[1:3] 
    Protein_Nucleus_graph, Protein_Cytoplasm_graph, Protein_Boundary_graph = graphs[4:6]

    du .= 0

    Gene_M = @view u[:, (Gene_Cell-1)*2+1]
    Gene_P = @view u[:, (Gene_Cell-1)*2+2]

    # Part A in Eq.(6)
    du[:, (Gene_Cell-1)*2+1] += k*GNNn(rep_intra1([Gene_M; Gene_P])) * Gene_M

    for i in 1:VT
        Mi = @view u[:, (i - 1)*2 + 1]
        Pi = @view u[:, (i - 1)*2 + 2]

        if i in Nucleus_Cells
            for j in mRNA_Nucleus_graph[i]
                Mj = @view u[:, (j - 1)*2 + 1]
                # #N^i_n * D^n_M * B * P_Mi in Eq.(6)
                du[:, (i - 1)*2 + 1] += (Dn1 * 𝐁) * Mi
                # Part B in Eq.(6)
                du[:, (j - 1)*2 + 1] += (Dn1 * GNNn(rep_Minter([Mj; Mi]))) * Mj
            end

            for j in Protein_Nucleus_graph[i]
                Pj = @view u[:, (j - 1)*2 + 2]
                # #N^i_n * D^n_P * B * P_Pi in Eq.(6)
                du[:, (i - 1)*2 + 2] += (Dn2 * 𝐁) * Pi
                # Part D in Eq.(6)
                du[:, (j - 1)*2 + 2] += (Dn2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj
            end

        elseif i in Cytoplasm_Cells
            # #N^i_c * D^c_M * B * P_Mi in Eq.(7)
            du[:, (i - 1)*2 + 1] += (dc1 * 𝐁) * Mi
            # #N^i_c * D^c_P * B * P_Pi in Eq.(7)
            du[:, (i - 1)*2 + 2] += (dc2 * 𝐁) * Pi
            # Part J in Eq.(7)
            du[:, (i - 1)*2 + 2] += (λ*GNNn(rep_intra2([Pi; Mi]))) * Pi

            for j in mRNA_Cytoplasm_graph[i]
                Mj = @view u[:, (j - 1)*2 + 1]
                # d_M * B * P_Mi in Eq.(7)
                du[:, (i - 1)*2 + 1] += (Dc1 * 𝐁) * Mi
                # Part G in Eq.(7)
                du[:, (j - 1)*2 + 1] += (Dc1 * GNNn(rep_Minter([Mj; Mi]))) * Mj
            end

            for j in Protein_Cytoplasm_graph[i]
                Pj = @view u[:, (j - 1)*2 + 2]
                # d_P * B * P_Pi in Eq.(7)
                du[:, (i - 1)*2 + 2] += (Dc2 * 𝐁) * Pi
                # Part K in Eq.(7)
                du[:, (j - 1)*2 + 2] += (Dc2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj
            end
        end

        if i in Nucleus_Boundary_Cells
            for j in mRNA_Boundary_graph[i]
                Mj = @view u[:, (j - 1)*2 + 1]
                # #N^i_c * D^nc_M * B * P_Mi in Eq.(6)
                du[:, (i - 1)*2 + 1] += (Dnc1 * 𝐁) * Mi
                # Part H in Eq.(7)
                du[:, (j - 1)*2 + 1] += (Dnc1 * GNNn(rep_Minter([Mj; Mi]))) * Mj
            end
        end

        if i in Cytoplasm_Boundary_Cells
            for j in Protein_Boundary_graph[i]
                Pj = @view u[:, (j - 1)*2 + 2]
                # #N^i_c * D^nc_P * B * P_Pi in Eq.(6) && #N^i_n * D^cn_P * B * P_Pi in Eq.(7)
                du[:, (i - 1)*2 + 2] += (Dnc2 * 𝐁) * Pi
                # Part E in Eq.(6) && part L in Eq.(7)
                du[:, (j - 1)*2 + 2] += (Dnc2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj
            end
        end
    end
end

# Read diffusion graph and Topology
include("./whole_cell_graph_VC_$(VT).jl");
Cells_sets = [Gene_Cell, Nucleus_Cells, Cytoplasm_Cells, Nucleus_Boundary_Cells, Cytoplasm_Boundary_Cells]  # Index of cells belonging to different parts
graphs = [mRNA_Nucleus_graph, mRNA_Cytoplasm_graph, mRNA_Boundary_graph,
          Protein_Nucleus_graph, Protein_Cytoplasm_graph, Protein_Boundary_graph]

# Initialize the ODE solver
ode_tf = tf
ode_tpoints = 1:tstep:ode_tf
tspan = (0, ode_tf);
u0s = Array{Any, 1}(undef, length(data_path))
problems = Array{Any, 1}(undef, length(data_path))
params = Array{Any, 1}(undef, length(data_path))

for i = 1:length(data_path)
    VT = VTs[i]
    params[i] = Float64.(readdlm("$(data_path[i])/params.csv", ',')[2, :])
    u0 = ssa_proba[i][:, :, 1]
    u0s[i] = ssa_proba[i][:, :, 1]
    problems[i] = ODEProblem((du, u, p, t) -> WholeCell_CME!(du, u, p, t; VT=VT, Cells_sets=Cells_sets, graphs=graphs), u0, tspan, params[i])
end

# Solve the ODE
idx = 1
version = 1
@time sol = Array(solve(problems[idx], Tsit5(), p = params[1], saveat=tstep));

ssa_Ms = cat([ssa_proba[idx][:, i, :] for i in 1:2:(2*VT)]..., dims=3);
ssa_Ps = cat([ssa_proba[idx][:, i, :] for i in 2:2:(2*VT)]..., dims=3);
sol_Ms = cat([sol[:, i, :] for i in 1:2:(2*VT)]..., dims=3);
sol_Ps = cat([sol[:, i, :] for i in 2:2:(2*VT)]..., dims=3);

# Plot the whole cell distribution
fig_colors = reshape(palette(:tab10)[:], 1, :);
t = 41
plt_idx = [1:6; 12:17; 23:28; 34:39; 45:50; 56:61] 

fig_xticks = [false for _ in 1:VT]
fig_xticks[56:61] .= true
fig_yticks = [false for _ in 1:VT]
fig_yticks[1:11:56] .= true

fig_colors = Array{Any, 1}(undef, VT)
for i in 1:VT
    if i == Gene_Cell
        fig_colors[i] = palette(:tab10)[1]
    elseif i in Nucleus_Cells
        fig_colors[i] = palette(:tab10)[2]
    elseif i in Cytoplasm_Cells
        fig_colors[i] = palette(:tab10)[3]
    end
end

mRNA_fig_labels = reshape(["mRNA $idx" for idx in 1:VT], 1, :)
Protein_fig_labels = reshape(["Protein $idx" for idx in 1:VT], 1, :)
mRNA_Figures = Any[]
Protein_Figures = Any[]

for i in plt_idx
    mRNA_subfig = plot(0:N, sol_Ms[:, t, i], c=fig_colors[i], label=mRNA_fig_labels[i],
                       xlims=(0, 30), ylims=(0, 1.0), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    mRNA_subfig = plot!(0:N, ssa_Ms[:, t, i], c=fig_colors[i], st=:scatter, ms=5)
    push!(mRNA_Figures, mRNA_subfig)

    Protein_subfig = plot(0:N, sol_Ps[:, t, i], c=fig_colors[i], label=Protein_fig_labels[i],
                          xlims=(0, 30), ylims=(0, 0.5), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    Protein_subfig = plot!(0:N, ssa_Ps[:, t, i], c=fig_colors[i], st=:scatter, ms=5)
    push!(Protein_Figures, Protein_subfig)
end

plot(mRNA_Figures..., layout=grid(6, 6), size=(1200, 1000), xlims=(0, 30), ylims=(0, 1.0), ms=4)
# savefig("Results/FigS4.pdf")

plot(Protein_Figures..., layout=grid(6, 6), size=(1200, 1000), xlims=(0, 30), ylims=(0, .5), ms=4)
# savefig("Results/FigS5.pdf")