# Import packages
using Flux, DifferentialEquations, Zygote
using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)

# Truncation
N = 39;  

# Model initialization
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
p = p_inter

# Load training parameters
@load "Fig2a-c/p.bson" p
ps = Flux.params(p);

# Define Hill function
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

# Define matrix ρ_i*A + d_i*B in Eq.(4)
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

    # Define matrix C_h in Eq.(4)
    C = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))
    for m = 1:length(D)  
        P = @view u[:, m]
        du[:, m] = (D[m] + length(Graph[m]) * C) * P
        for j in Graph[m]
            P_neighbor = @view u[:, j]
                        
            # Define NN^{j->i}_θ in Eq.(4)
            NN_in = rep_inter([P; P_neighbor])
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)
            du[:, m] += G_in * P
        end
    end
end

# Number of cells
VT = 16

# Select topology (letter "E")
grids = "E"

# Load distribution data (SSA)
ssa_path = "Fig2a-c/data"
test_ssa_path = "$(ssa_path)/$(VT)_cells_$grids"
proba_list = [reshape(readdlm("$(test_ssa_path)/proba/cell_$(v).csv", ','), (N+1, 1, 201)) for v in 1:VT]  
test_ssa_proba = cat(proba_list..., dims=2)

# Load kinetic parameter file
params = readdlm("$(test_ssa_path)/params.csv", ',')
rho = Float64.(params[2:end, 1])
d = Float64.(params[2:end, 2])

# Initialize the ODE solver
tf = 20.0; 
tspan = (0, tf);  
tstep = 0.1; 
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = Dict(); @load "$(test_ssa_path)/graph.bson" graph
test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);

# Solve the ODE
@time sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));

# Plot cells and snapshots of interest
fig_colors = reshape(palette(:glasbey_hv_n256)[:], 1, :);
fig_labels = reshape(["cell $i" for i in 1:VT], 1, VT)

figures = Any[];
fig_xticks = [[false for _ in 1:12]; [true for _ in 1:4]]
fig_yticks = vcat([[true; false; false; false] for _ in 1:4]...)
for i in 1:16
    subfig = plot(xlims=(0, 30), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    subfig = plot!(sol[1:30, i, end], c=fig_colors[16*i], label=join([fig_labels[i]," GNN-MME"]),legend=:topright);
    subfig = plot!(test_ssa_proba[1:30, i, end], st=:scatter, ms=5, c=fig_colors[16*i],label=join([fig_labels[i]," SSA"]));
    push!(figures, subfig)
end
f = plot(figures..., layout = grid(4, 4), size = (800, 600))
# savefig("Results/Fig2bc_VT=$(VT)_$grids.pdf")
