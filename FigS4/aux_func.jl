using StatsBase, Distributions

# Get the index of the i-th cell
get_index(i; N) = ((i - 1) * (N + 1) + 1):(i * (N + 1))  
get_index(i) = get_index(i, N=N)

# Calculate mean value according to the distribution P
P2mean(P) = [P[i] * (i-1) for i in 1:length(P)] |> sum

# Calculate variance var
P2var(P) = ([P[i] * (i-1)^2 for i in 1:length(P)] |> sum) - P2mean(P)^2

# Calculate second moment sm
P2sm(P) = [P[i] * (i-1)^2 for i in 1:length(P)] |> sum

function traje2proba(t; N)
    @assert length(size(t)) == 1
    bins = collect(0:N+1)
    hist = fit(Histogram, t, bins)
    h_den = hist.weights / length(t)
    h_den
end
