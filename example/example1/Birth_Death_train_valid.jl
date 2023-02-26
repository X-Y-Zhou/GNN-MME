## import package
using Flux, DifferentialEquations, Zygote
using DiffEqSensitivity
using Distributions, Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @save, @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
N = 39;  # truncation

## Load training set
ssa_path = "example/example1/data"
train_path = "example/example1"
data_path = vcat(["$ssa_path/2_cells_v$(i)" for i in 1:3]...)
VTs = [2, 2, 2]
ssa_proba = Array{Any, 1}(undef, length(data_path))

Threads.@threads for i in 1:length(VTs)
    proba_list = [reshape(readdlm("$(data_path[i])/proba/cell_$(v).csv", ','), (N+1, 1, 201)) for v in 1:VTs[i]]
    ssa_proba[i] = cat(proba_list..., dims=2)
end
ssa_proba[1]  # 维度为 [N, VT, timestamp]
## model destructure define
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
@load "$train_path/p.bson" p  # 加载训练参数
ps = Flux.params(p);
## define hill function and CME
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

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

function CME!(du, u, p, t; Graph, D)
    rep_inter = re_inter(p)    # GNNin 网络重构
    # hill function Difussion 的外传矩阵
    B = spdiagm(0 => [0.0; -hill(collect(1.0:N))], 1 => hill(collect(1.0:N)))
    for m = 1:length(D)  # 遍历每一个细胞
        P = @view u[:, m]
        du[:, m] = (D[m] + length(Graph[m]) * B) * P
        for j in Graph[m]
            P_neighbor = @view u[:, j]  # 邻居细胞分布
            NN_in = rep_inter([P; P_neighbor])  # 网络的输出
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)  # 转化成矩阵
            du[:, m] += G_in * P
        end
    end
end

#
tf = 12.0;  # 定义问题求解的最长时间
tspan = (0, tf);  # 问题求解范围
tstep = 0.1;
# u0s = Array{Any, 1}(undef, length(data_path))
graphs = Array{Any, 1}(undef, length(data_path))  # 保存拓扑结构
Ds = Array{Any, 1}(undef, length(data_path))  # 常矩阵部分
problems = Array{Any, 1}(undef, length(data_path))
for i = 1:length(data_path)
    VT = VTs[i]  # 细胞数量
    params = readdlm("$(data_path[i])/params.csv", ',')  # 读取参数文件
    rho = Float64.(params[2:end, 1])
    d = Float64.(params[2:end, 2])
    Ds[i] = sparse_D(rho, d)
    u0 = zeros(N+1, VT)  # 初始分布
    u0[1, :] .= 1

    graph = Dict(); @load "$(data_path[i])/graph.bson" graph  # 读取拓扑结构
    graphs[i] = graph
    problems[i] = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graphs[i], D=Ds[i]), u0, tspan, p)
end
problems;


set = 1
sol = Array(solve(problems[set], Tsit5(), p = ps[1], saveat=tstep))

sol1 = sol[:,1,:]
sol2 = sol[:,2,:]

ssa_proba[i][1:30,1,end]
sol1[1:30,end]

fig_colors = reshape(palette(:tab10)[:], 1, :);
fig_labels = reshape(["cell $i" for i in 1:VTs[i]], 1, VTs[i])

figures = Any[];
fig_xticks = [false,true]
fig_yticks = [true,true]

for i in 1:2
    subfig = plot(xlims=(0, 20), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    subfig = plot!(sol[:,i,:][1:30,16], c=fig_colors[i], label=fig_labels[i]);
    subfig = plot!(ssa_proba[set][1:30,i,16], st=:scatter, ms=5, c=fig_colors[i])
    push!(figures, subfig)

    subfig = plot(xlims=(0, 20), ylims=(0, 0.3), xticks=fig_xticks[i], yticks=fig_yticks[i], grids=false)
    subfig = plot!(sol[:,i,:][1:30,end], c=fig_colors[i], label=fig_labels[i]);
    subfig = plot!(ssa_proba[set][1:30,i,end], st=:scatter, ms=5, c=fig_colors[i])
    push!(figures, subfig)
end

plot(figures..., layout = grid(2, 2), size = (800, 600))

#writedlm("$(ssa_path)/2_cells_v$i/proba/cell_1_pred.csv",sol1,',')
#writedlm("$(ssa_path)/2_cells_v$i/proba/cell_2_pred.csv",sol2,',')