using StatsBase, Distributions

get_index(i; N) = ((i - 1) * (N + 1) + 1):(i * (N + 1)) # 获得第 i 个细胞的索引
get_index(i) = get_index(i, N=N)
# get_cells(v) = vcat([get_index(i) for i in v]...)    # 获得向量 v 中所有细胞的索引
P2mean(P) = [P[i] * (i-1) for i in 1:length(P)] |> sum  # 根据分布 P 计算均值 mean
P2var(P) = ([P[i] * (i-1)^2 for i in 1:length(P)] |> sum) - P2mean(P)^2 # 根据分布 P 计算方差 var
P2sm(P) = [P[i] * (i-1)^2 for i in 1:length(P)] |> sum  # 根据分布 P 计算二阶矩 sm

function traje2proba(t; N)
    @assert length(size(t)) == 1
    bins = collect(0:N+1)
    hist = fit(Histogram, t, bins)
    h_den = hist.weights / length(t)
    h_den
end
