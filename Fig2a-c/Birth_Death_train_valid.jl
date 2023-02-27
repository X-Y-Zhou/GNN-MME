# Import package
using Flux, DifferentialEquations, Zygote
using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @save, @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)

# Truncation
N = 39;

# Load training set
ssa_path = "Fig2a-c/data"
data_path = vcat(["$ssa_path/2_cells_v$(i)" for i in 1:3]...)
VTs = [2, 2, 2]
ssa_proba = Array{Any, 1}(undef, length(data_path))

Threads.@threads for i in 1:length(VTs)
    proba_list = [reshape(readdlm("$(data_path[i])/proba/cell_$(v).csv", ','), (N+1, 1, 201)) for v in 1:VTs[i]]
    ssa_proba[i] = cat(proba_list..., dims=2)
end

# Model initialization
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);

# Load training parameters
@load "Fig2a-c/p.bson" p  
ps = Flux.params(p);

# Define Hill function
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)
B = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))

# Define Di matrix
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
    C = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))
    for m = 1:length(D)
        P = @view u[:, m]
        du[:, m] = (D[m] + length(Graph[m]) * C) * P
        for j in Graph[m]
            P_neighbor = @view u[:, j]
            NN_in = rep_inter([P; P_neighbor])
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)
            du[:, m] += G_in * P
        end
    end
end

# Initialize the ODE solver and Read parameter file and topology
tf = 12.0;
tspan = (0, tf);
tstep = 0.1;

graphs = Array{Any, 1}(undef, length(data_path))
Ds = Array{Any, 1}(undef, length(data_path))
problems = Array{Any, 1}(undef, length(data_path))

for i = 1:length(data_path)
    VT = VTs[i] 
    params = readdlm("$(data_path[i])/params.csv", ',')
    rho = Float64.(params[2:end, 1])
    d = Float64.(params[2:end, 2])
    Ds[i] = sparse_D(rho, d)
    u0 = zeros(N+1, VT)
    u0[1, :] .= 1

    graph = Dict(); @load "$(data_path[i])/graph.bson" graph
    graphs[i] = graph
    problems[i] = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graphs[i], D=Ds[i]), u0, tspan, p)
end

# Solve the ODE
set = 1
sol = Array(solve(problems[set], Tsit5(), p = ps[1], saveat=tstep))

#Plot the selected cell and snapshots
fig_colors = reshape(palette(:tab10)[:], 1, :);
fig_labels = ["cell1","cell2"]

figures = Any[];
fig_xticks = [false,true]
fig_yticks = [true,true]

for i in 1:2
    subfig = plot(xlims=(0, 20), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false,title="t=1.5")
    subfig = plot!(sol[:,i,:][1:30,16], c=fig_colors[i], label=join([fig_labels[i]," GNN-MME"]));
    subfig = plot!(ssa_proba[set][1:30,i,16], st=:scatter, ms=5, c=fig_colors[i],label=join([fig_labels[i]," SSA"]))
    push!(figures, subfig)

    subfig = plot(xlims=(0, 20), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false,title="t=12")
    subfig = plot!(sol[:,i,:][1:30,end], c=fig_colors[i], label=join([fig_labels[i]," GNN-MME"]));
    subfig = plot!(ssa_proba[set][1:30,i,end], st=:scatter, ms=5, c=fig_colors[i],label=join([fig_labels[i]," SSA"]))
    push!(figures, subfig)
end

plot(figures..., layout = grid(2, 2), size = (800, 600))
#savefig("Results/Fig2a_set1.pdf")