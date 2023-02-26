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
N = 39;  # truncation

## Load training set
VT = 121
ssa_path = "FigureS3/data"
data_path = ["$ssa_path/$(VT)_cells"]
VTs = fill(VT, length(data_path))
ssa_proba = Array{Any, 1}(undef, length(data_path))

tf = 150  # ODE solver end time
tstep = 0.5  # Sampling interval
# tpoints = 0:tstep:tf
tpoints = readdlm("$(data_path[1])/timepoints.csv", ',')[:]  # All time points
tstamp = length(tpoints)  # Number of time points

Threads.@threads for i in 1:length(VTs)
    mRNA_proba_list = [reshape(readdlm("$(data_path[i])/proba/M_$(v).csv", ','), (N+1, 1, tstamp)) for v in 1:VTs[i]]
    Protein_proba_list = [reshape(readdlm("$(data_path[i])/proba/P_$(v).csv", ','), (N+1, 1, tstamp)) for v in 1:VTs[i]]
    # ssa_proba[i] = cat(mRNA_proba_list..., Protein_proba_list..., dims=2)
    ssa_proba[i] = hcat([hcat(mRNA_proba_list[j], Protein_proba_list[j]) for j in 1:VTs[i]]...)
end
ssa_proba[1]  # Dimension is [N, VT, tpoints]; M1, P1, M2, P2, ...

## model destructure define
hill(n::Union{Int64, Float64}; k=10.0, K=5.0) = k / (n + K)
hill(ns::Vector; k=10.0, K=5.0) = [hill(ns[i], k=k, K=K) for i in 1:length(ns)]

train_path = "FigureS3/model_params"
# GNNintra1
intra1_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                     x -> leakyrelu.(x))
p_intra1, re_intra1 = Flux.destructure(intra1_struc)
@load "$train_path/p_intra1.bson" p_intra1
rep_intra1 = re_intra1(p_intra1)

# GNNintra2
intra2_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                     x -> leakyrelu.(x))
p_intra2, re_intra2 = Flux.destructure(intra2_struc)
@load "$train_path/p_intra2.bson" p_intra2
rep_intra2 = re_intra2(p_intra2)

# mRNA inter network
Minter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                       x -> leakyrelu.(x))
p_Minter, re_Minter = Flux.destructure(Minter_struc)
@load "$train_path/p_Minter.bson" p_Minter
rep_Minter = re_Minter(p_Minter)

# Protein inter network
Pinter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N),
                       x -> leakyrelu.(x))
p_Pinter, re_Pinter = Flux.destructure(Pinter_struc)
@load "$train_path/p_Pinter.bson" p_Pinter
rep_Pinter = re_Minter(p_Pinter)

## define CME
𝐀 = spdiagm(0 => -collect(0:N)) + spdiagm(1 => collect(1:N))  # degraded upper triangular matrix
𝐁 = spdiagm(0 => [-ones(N); 0.0]) + spdiagm(-1 => ones(N))    # generate lower triangular matrix
GNNn(v) = spdiagm(0 => [-v; 0.0]) + spdiagm(-1 => v)  # Convert the output of the network to a matrix

function WholeCell_CME!(du, u, p, t; VT, Cells_sets, graphs)
    # Get the parameters of the cell
    k, K, λ, dn1, dn2, dc1, dc2 = p[1:7]
    Dn1, Dn2, Dc1, Dc2, Dnc1, Dnc2 = p[8:13]
    Gene_Cell = Cells_sets[1]  # The grid where Gene is located
    Nucleus_Cells, Cytoplasm_Cells = Cells_sets[2:3] # Grid belonging to cell nucleus and cytoplasm
    Nucleus_Boundary_Cells, Cytoplasm_Boundary_Cells = Cells_sets[4:5]  # Grid on the border
    mRNA_Nucleus_graph, mRNA_Cytoplasm_graph, mRNA_Boundary_graph = graphs[1:3]  # Hash table of mRNA diffusion
    Protein_Nucleus_graph, Protein_Cytoplasm_graph, Protein_Boundary_graph = graphs[4:6]  # Hash table of Protein Diffusion

    du .= 0

    # Gene
    Gene_M = @view u[:, (Gene_Cell-1)*2+1]
    Gene_P = @view u[:, (Gene_Cell-1)*2+2]
    du[:, (Gene_Cell-1)*2+1] += k*GNNn(rep_intra1([Gene_M; Gene_P])) * Gene_M  # G -> G + Mi
    # Transcription without Feedback
    # du[:, (Gene_Cell-1)*2+1] += k*𝐁 * Gene_M  # G -> G + Mi

    for i in 1:VT
        Mi = @view u[:, (i - 1)*2 + 1]
        Pi = @view u[:, (i - 1)*2 + 2]
        if i in Nucleus_Cells  # If i is the nucleus
            for j in mRNA_Nucleus_graph[i]  # Nucleus Diffusion of mRNA
                Mj = @view u[:, (j - 1)*2 + 1]
                du[:, (i - 1)*2 + 1] += (Dn1 * 𝐀) * Mi  # Dn1: Mi -> Mj
                du[:, (j - 1)*2 + 1] += (Dn1 * GNNn(rep_Minter([Mj; Mi]))) * Mj  # Dn1: Mi -> Mj
            end

            for j in Protein_Nucleus_graph[i]  # Nucleus Diffusion Protein
                Pj = @view u[:, (j - 1)*2 + 2]
                du[:, (i - 1)*2 + 2] += (Dn2 * 𝐀) * Pi  # Dn2: Pi -> Pj
                du[:, (j - 1)*2 + 2] += (Dn2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj  # Dn2: Pi -> Pj
            end
        elseif i in Cytoplasm_Cells  # if i is cytoplasmic
            du[:, (i - 1)*2 + 1] += (dc1 * 𝐀) * Mi  # dc1: Mi -> ∅
            du[:, (i - 1)*2 + 2] += (dc2 * 𝐀) * Pi  # dc2: Pi -> ∅
            du[:, (i - 1)*2 + 2] += (λ*GNNn(rep_intra2([Pi; Mi]))) * Pi  # λ: Mi -> Mi + Pi

            for j in mRNA_Cytoplasm_graph[i]  # Cytoplasm Diffusion of mRNA
                Mj = @view u[:, (j - 1)*2 + 1]
                du[:, (i - 1)*2 + 1] += (Dc1 * 𝐀) * Mi  # Dc1: Mi -> Mj
                du[:, (j - 1)*2 + 1] += (Dc1 * GNNn(rep_Minter([Mj; Mi]))) * Mj  # Dc1: Mi -> Mj
            end

            for j in Protein_Cytoplasm_graph[i]  #Cytoplasm Diffusion of Protein
                Pj = @view u[:, (j - 1)*2 + 2]
                du[:, (i - 1)*2 + 2] += (Dc2 * 𝐀) * Pi  # Dc2: Pi -> Pj
                du[:, (j - 1)*2 + 2] += (Dc2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj  # Dc2: Pi -> Pj
            end
        end

        if i in Nucleus_Boundary_Cells  # If grid i is on the border of the nucleus (mRNA diffuse unidirectionally)
            for j in mRNA_Boundary_graph[i]  # Boundary Diffusion of mRNA
                Mj = @view u[:, (j - 1)*2 + 1]
                du[:, (i - 1)*2 + 1] += (Dnc1 * 𝐀) * Mi
                du[:, (j - 1)*2 + 1] += (Dnc1 * GNNn(rep_Minter([Mj; Mi]))) * Mj
            end
        end

        if i in Cytoplasm_Boundary_Cells  # If grid i is on the cytoplasmic boundary (Protein diffuse bidirectionally)
            for j in Protein_Boundary_graph[i]  # Boundary Diffusion of Protein
                Pj = @view u[:, (j - 1)*2 + 2]
                du[:, (i - 1)*2 + 2] += (Dnc2 * 𝐀) * Pi
                du[:, (j - 1)*2 + 2] += (Dnc2 * GNNn(rep_Pinter([Pj; Pi]))) * Pj
            end
        end
    end
end

##
include("./whole_cell_graph_VC_$(VT).jl");  # Diffusion graph of Protein and mRNA
Cells_sets = [Gene_Cell, Nucleus_Cells, Cytoplasm_Cells, Nucleus_Boundary_Cells, Cytoplasm_Boundary_Cells]  # Index of cells belonging to different parts
graphs = [mRNA_Nucleus_graph, mRNA_Cytoplasm_graph, mRNA_Boundary_graph,
          Protein_Nucleus_graph, Protein_Cytoplasm_graph, Protein_Boundary_graph]  # Topology
ode_tf = tf  # ODE solver end time 
ode_tpoints = 1:tstep:ode_tf  # Time point
tspan = (0, ode_tf);  # Time range
u0s = Array{Any, 1}(undef, length(data_path))
problems = Array{Any, 1}(undef, length(data_path))
params = Array{Any, 1}(undef, length(data_path))

for i = 1:length(data_path)
    VT = VTs[i]
    params[i] = Float64.(readdlm("$(data_path[i])/params.csv", ',')[2, :])  # Read parameter file
    u0 = ssa_proba[i][:, :, 1]  # [M1, P1, M2, P2, ...]
    u0s[i] = ssa_proba[i][:, :, 1]
    problems[i] = ODEProblem((du, u, p, t) -> WholeCell_CME!(du, u, p, t; VT=VT, Cells_sets=Cells_sets, graphs=graphs), u0, tspan, params[i])
end
length(problems)

## Validation of training results
idx = 1
version = 1
@time sol = Array(solve(problems[idx], Tsit5(), p = params[1], saveat=tstep));

ssa_Ms = cat([ssa_proba[idx][:, i, :] for i in 1:2:(2*VT)]..., dims=3);
ssa_Ps = cat([ssa_proba[idx][:, i, :] for i in 2:2:(2*VT)]..., dims=3);
sol_Ms = cat([sol[:, i, :] for i in 1:2:(2*VT)]..., dims=3);
sol_Ps = cat([sol[:, i, :] for i in 2:2:(2*VT)]..., dims=3);

##Visualize all grids
fig_colors = reshape(palette(:tab10)[:], 1, :);
t = 41 # time = 20s
# Plot the distribution of the whole cell
plt_idx = [1:6; 12:17; 23:28; 34:39; 45:50; 56:61]  # 11 * 11
# The plots sharing the axis parameters (the axis in the middle part is not displayed)
fig_xticks = [false for _ in 1:VT]
fig_xticks[56:61] .= true
fig_yticks = [false for _ in 1:VT]
fig_yticks[1:11:56] .= true

# Different parts are distinguished by different colors
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
#savefig("FigureS3/results/Figure_S3.pdf")

plot(Protein_Figures..., layout=grid(6, 6), size=(1200, 1000), xlims=(0, 30), ylims=(0, .5), ms=4)
#savefig("FigureS3/results/Figure_S4.pdf")

## Plot mean value
fig_colors = reshape(palette(:tab10)[:], 1, :);
plt_cells = [1, 13, 25, 37, 49, 61]  # 11 * 11
ssa_mRNA_Mean = hcat([[P2mean(ssa_Ms[:, t, c]) for t in 1:length(tpoints)] for c in 1:VT]...)
sol_mRNA_Mean = hcat([[P2mean(sol_Ms[:, t, c]) for t in 1:length(tpoints)] for c in 1:VT]...)

ssa_Protein_Mean = hcat([[P2mean(ssa_Ps[:, t, c]) for t in 1:length(tpoints)] for c in 1:VT]...)
sol_Protein_Mean = hcat([[P2mean(sol_Ps[:, t, c]) for t in 1:length(tpoints)] for c in 1:VT]...)
# cell_idx = 61
Figures = Any[]
for (i, cell_idx) in enumerate(plt_cells)
    fig = plot(xlims=(0, tpoints[end]), ylims=(0, 18), title="Mean");
    fig = plot!(tpoints[1:10:end], ssa_mRNA_Mean[1:10:end, cell_idx], st=:scatter, c=fig_colors[1], ms=3);
    fig = plot!(tpoints, sol_mRNA_Mean[1:end, cell_idx], c=fig_colors[1], label="mRNA Cell $(cell_idx)")

    fig = plot!(tpoints[1:10:end], ssa_Protein_Mean[1:10:end, cell_idx], st=:scatter, c=fig_colors[2], ms=3);
    fig = plot!(tpoints, sol_Protein_Mean[:, cell_idx], c=fig_colors[2], label="Protein Cell $(cell_idx)")

    push!(Figures, fig)
end
plot(Figures..., layout=grid(2, 3), size=(800, 400))