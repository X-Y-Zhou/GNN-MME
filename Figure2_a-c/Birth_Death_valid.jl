## import package
using Flux, DifferentialEquations, Zygote
using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
N = 39;  # truncation

# model destructure define
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
p = p_inter
@load "Figure2_a-c/p.bson" p  # Load training parameters
ps = Flux.params(p);
## define hill function and CME
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

function sparse_D(ρ, d)  # Generate Di matrix
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

function CME!(du, u, p, t; Graph, D)
    rep_inter = re_inter(p)    # GNNin network reconstruction
    # External transmission matrix of hill function Difussion
    B = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))
    for m = 1:length(D)  # Iterate through each cell
        P = @view u[:, m]
        du[:, m] = (D[m] + length(Graph[m]) * B) * P
        for j in Graph[m]  # Consider the Intercellar part of each neighbor cell
            P_neighbor = @view u[:, j]  # Distribution of neighbor cell
            NN_in = rep_inter([P; P_neighbor])  # Output of network
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)  # Convert to matrix
            du[:, m] += G_in * P
        end
    end
end

##
tf = 20.0;  # ODE solver end time
tspan = (0, tf);  # Time range
tstep = 0.1; # Sampling time step
ssa_path = "Figure2_a-c/data"

VT = 16  # Number of cells
grids = "E" # grids

### attention!!!!
test_ssa_path = "$(ssa_path)/$(VT)_cells_$grids" # Path of test data
proba_list = [reshape(readdlm("$(test_ssa_path)/proba/cell_$(v).csv", ','), (N+1, 1, 201)) for v in 1:VT]  # Read distribution data
test_ssa_proba = cat(proba_list..., dims=2)
params = readdlm("$(test_ssa_path)/params.csv", ',')  # Read parameter file
rho = Float64.(params[2:end, 1])
d = Float64.(params[2:end, 2])

u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = Dict(); @load "$(test_ssa_path)/graph.bson" graph
test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
@time sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));

#=
for i = 1 : VT
    solt = sol[:,i,:]
    writedlm("$(test_ssa_path)/proba/cell_$(i)_pred.csv",solt,',')
end
=#

fig_colors = reshape(palette(:glasbey_hv_n256)[:], 1, :)
fig_labels = reshape(["cell $i" for i in 1:VT], 1, VT)

#
figures = Any[];
fig_xticks = [[false for _ in 1:12]; [true for _ in 1:4]]
fig_yticks = vcat([[true; false; false; false] for _ in 1:4]...)
for i in 1:16
    subfig = plot(xlims=(0, 30), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    subfig = plot!(sol[1:30, i, end], c=fig_colors[16*i], label=fig_labels[i]);
    subfig = plot!(test_ssa_proba[1:30, i, end], st=:scatter, ms=5, c=fig_colors[16*i]);
    push!(figures, subfig)
end
f = plot(figures..., layout = grid(4, 4), size = (800, 600))
savefig("Figure2_a-c/results/Figure2bc_VT=$(VT)_$grids.pdf")
