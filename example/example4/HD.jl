# 创建 Neural-ODE 的 Hellinger Dist 表格
using Flux, DifferentialEquations
using Distances
using DelimitedFiles, LinearAlgebra, StatsBase, Statistics
using BSON: @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
include("aux_functions.jl")
N = 39;  # truncation

## 定义网络结构
path = "./example/example4"
scale_list = [2, 2.5, 3, 3.5, 4, 4.5]  # 考虑的尺度
sample_size_list = @. Int(round(10 ^ scale_list))
# model destructure define
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
p = p_inter

## 定义矩阵与 ODE 
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

function sparse_D(ρ, d)  # 生成 Di 矩阵
    M = length(ρ)
    # D = zeros(N + 1, N + 1, M)
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
        for j in Graph[m]  # 考虑每一个邻居的 Intercellular 部分
            P_neighbor = @view u[:, j]
            NN_in = rep_inter([P; P_neighbor])
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)
            du[:, m] += G_in * P
        end
    end
end


## 10 Cells
VT = 10  # 细胞数量
tf = 20.0;  # ODE 求解的最终时刻
tspan = (0, tf);  # ODE 求解的区间

# 3 次独立重复实验
p_path_list = [
    ["p_scale_2.0_v1.bson", "p_scale_2.0_v2.bson", "p_scale_2.0_v3.bson"],
    ["p_scale_2.5_v1.bson", "p_scale_2.5_v2.bson", "p_scale_2.5_v3.bson"],
    ["p_scale_3.0_v1.bson", "p_scale_3.0_v2.bson", "p_scale_3.0_v3.bson"],
    ["p_scale_3.5_v1.bson", "p_scale_3.5_v2.bson", "p_scale_3.5_v3.bson"],
    ["p_scale_4.0_v1.bson", "p_scale_4.0_v2.bson", "p_scale_4.0_v3.bson"],
    ["p_scale_4.5_v1.bson", "p_scale_4.5_v2.bson", "p_scale_4.5_v3.bson"],
]

# ρ=2.5, d=0.5 参数的表
table_list = [[0. for _ in 1:3] for _ in 1:6]
SS_proba_list_v1 = Array{Any, 2}(undef, 6, 3)  # 稳态分布
version = 1
tstep = 0.1; # 求解结果的采样时间步长
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]  # 10^7 得到的 SSA 分布
rho = [2.5 for _ in 1:VT]
d = [0.5 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)  # 环形拓扑结构
for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        @load "$path/model_params/$(p_path)" p  # 加载训练参数

        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v1[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), ssa_SS_P)
    end
end
table_v1 = hcat(table_list...)

# ρ=1.0, d=1.0 参数的表（extrapolate)
table_list = [[0. for _ in 1:3] for _ in 1:6]
SS_proba_list_v2 = Array{Any, 2}(undef, 6, 3)  # 稳态分布
version = 3
tstep = 0.1; # 求解结果的采样时间步长
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]  # 10^7 得到的 SSA 分布
rho = [1.0 for _ in 1:VT]
d = [1.0 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)  # 环形拓扑结构
for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        @load "$path/model_params/$(p_path)" p  # 加载训练参数

        # 求解 ODE 得到分布
        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v2[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), abs.(ssa_SS_P))
    end
end
table_v2 = hcat(table_list...)

## 画图
fig_colors = reshape(palette(:tab10)[:], 1, :);
# v1 的图
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


# extrapolate 的图
version = 3  # rho=1, d=1
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
# savefig("./Figures/example4/HD.pdf")


## 画出 Nueral-ODE 和 SSA 的对比图
Figures = Any[]
version = 1  # 两个图单独出
SSA_proba = [readdlm("example/example4/data/10_cells_v$(version)/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
# fig_ystick = [true; [false for _ in 1:5]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter)
    f = plot!(SS_proba_list_v1[i, 1])
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 30), ylims=(0, 0.25), size=(1100, 200))
# savefig("./example/example4/NeuralODE_SSA_compare_v$(version).pdf")

## 
Figures = Any[]
version = 3  # 两个图单独出
SSA_proba = [readdlm("example/example4/data/10_cells_v$(version)/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
# fig_ystick = [true; [false for _ in 1:5]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter)
    f = plot!(SS_proba_list_v2[i, 1])
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 20), ylims=(0, 0.50), size=(1100, 200))
# savefig("./example/example4/NeuralODE_SSA_compare_v$(version).pdf")


# 创建 Neural-ODE 的 Hellinger Dist 表格
using Flux, DifferentialEquations
using Distances
using DelimitedFiles, LinearAlgebra, StatsBase
using BSON: @load
using SparseArrays
using Plots
default(lw=3, msw=0., label=false)
include("aux_functions.jl")
N = 39;  # truncation

# tlb = readdlm("./examples/example4/data/10_cells/distance_table.csv", ',')
# mean(tlb, dims=1)

## 
path = "./example/example4"
scale_list = [2, 2.5, 3, 3.5, 4, 4.5]
sample_size_list = @. Int(round(10 ^ scale_list))
# model destructure define
inter_struc = Chain(Dense(2*(N + 1), 20, tanh), Dense(20, N, leakyrelu))
p_inter, re_inter = Flux.destructure(inter_struc);
p = p_inter

## 
hill(x; k = 10., K = 5.0) = @. k * x / (K + x)

function sparse_D(ρ, d)  # 生成 Di 矩阵
    M = length(ρ)
    # D = zeros(N + 1, N + 1, M)
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
        for j in Graph[m]  # 考虑每一个邻居的 Intercellular 部分
            P_neighbor = @view u[:, j]
            NN_in = rep_inter([P; P_neighbor])
            G_in = spdiagm(0 => [-NN_in; 0.0], -1 => NN_in)
            du[:, m] += G_in * P
        end
    end
end


## 10 Cells
VT = 10
tf = 20.0;  # ODE 求解的最终时刻
tspan = (0, tf);

p_path_list = [
    ["p_scale_2.0_v1.bson", "p_scale_2.0_v2.bson", "p_scale_2.0_v3.bson"],
    ["p_scale_2.5_v1.bson", "p_scale_2.5_v2.bson", "p_scale_2.5_v3.bson"],
    ["p_scale_3.0_v1.bson", "p_scale_3.0_v2.bson", "p_scale_3.0_v3.bson"],
    ["p_scale_3.5_v1.bson", "p_scale_3.5_v2.bson", "p_scale_3.5_v3.bson"],
    ["p_scale_4.0_v1.bson", "p_scale_4.0_v2.bson", "p_scale_4.0_v3.bson"],
    ["p_scale_4.5_v1.bson", "p_scale_4.5_v2.bson", "p_scale_4.5_v3.bson"],
]

# version 1 的表
table_list = [[0. for _ in 1:3] for _ in 1:6]
SS_proba_list_v1 = Array{Any, 2}(undef, 6, 3)
version = 1
tstep = 0.1; # 求解结果的采样时间步长
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]
rho = [2.5 for _ in 1:VT]
d = [0.5 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)  # 环形拓扑结构
for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        @load "$path/model_params/$(p_path)" p  # 加载训练参数

        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v1[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), ssa_SS_P)
    end
end
table_v1 = hcat(table_list...)
# writedlm("$path/DistTable/Neural_$(VT)_distance_table_v1.csv", table, ',')

# version 2 的表
table_list = [[0. for _ in 1:3] for _ in 1:6]
SS_proba_list_v2 = Array{Any, 2}(undef, 6, 3)
version = 2
tstep = 0.1; # 求解结果的采样时间步长
ssa_SS_P = readdlm("$path/data/$(VT)_cells_v$(version)/proba/cell_1.csv", ',')[:, end]
rho = readdlm("$path/data/$(VT)_cells_v$(version)/params.csv", ',')[2:end, 1]
d = readdlm("$path/data/$(VT)_cells_v$(version)/params.csv", ',')[2:end, 2]
# rho = [2.5 for _ in 1:VT]
# d = [0.5 for _ in 1:VT]
u0 = zeros(N+1, VT)
u0[1, :] .= 1
graph = circle_graph(VT)  # 环形拓扑结构
for i in 1:6
    for j in 1:3
        p_path = p_path_list[i][j]
        @load "$path/model_params/$(p_path)" p  # 加载训练参数

        test_prob = ODEProblem((du, u, p, t) -> CME!(du, u, p, t; Graph=graph, D=sparse_D(rho, d)), u0, tspan, p);
        sol = Array(solve(test_prob, Tsit5(), p=p, saveat=tstep));
        sol_SS_P = sol[:, 1, end]
        SS_proba_list_v2[i, j] = sol_SS_P
        table_list[i][j] = hellinger(abs.(sol_SS_P), abs.(ssa_SS_P))
    end
end
table_v2 = hcat(table_list...)

## 画图
fig_colors = reshape(palette(:tab10)[:], 1, :);
# v1 的图
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


# v2 的图
version = 2
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
# savefig("./Figures/example4/HD.pdf")


## 画出 Nueral-ODE 和 SSA 的对比图
Figures = Any[]
SSA_proba = [readdlm("example/example4/data/10_cells_v1/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
# fig_ystick = [true; [false for _ in 1:5]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    # f = plot(grids=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter)
    # f = plot!(SSA_proba[end], st=scatter)
    f = plot!(SS_proba_list_v1[i, 1])
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 30), ylims=(0, 0.25), size=(1100, 200))
# savefig("./Figures/example4/NeuralODE_SSA_compare_v1.pdf")

Figures = Any[]
SSA_proba = [readdlm("example/example4/data/10_cells_v2/sample/proba_scale_$(scale)_v1.csv", ',')[:, end] for scale in [scale_list; 7.0]]
# fig_ystick = [true; [false for _ in 1:5]]
for i in 1:6
    f = plot(title="scale $(scale_list[i])", grids=false, xticks=false)
    # f = plot(grids=false)
    if i > 1
        f = plot!(yticks=false)
    end
    f = plot!(SSA_proba[i], st=:scatter)
    # f = plot!(SSA_proba[end], st=scatter)
    f = plot!(SS_proba_list_v2[i, 1])
    push!(Figures, f)
end
plot(Figures[[1; 2; 3; 4; 5; 6]]..., layout=grid(1, 6), xlim=(0, 30), ylims=(0, 0.25), size=(1100, 200))

