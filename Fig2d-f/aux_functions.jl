function find_neighbour(row::Int,column::Int)
    vec_list = Array{Any,1}(undef,row*column)
    for i in 1:row*column
        if i%row == 1
            vec_list[i] = [i+1, i-row, i+row]
        elseif i%row == 0
            vec_list[i] = [i-1, i-row, i+row]
        else
            vec_list[i] = [i-1, i+1, i-row, i+row]
        end
        filter!(x->x.>0&&x.<=row*column,vec_list[i])
    end
    return vec_list
end


function pfsample(w,s,n::Int64)
    threshold = rand() * s
    i = 1
    cw = w[1]
    while cw < threshold && i < n
        i += 1
        cw += w[i]
    end
    return i
end

function component(reactants::Vector)
    n_react = size(reactants[1],2)
    numofRun = length(reactants)
    reactants_list = Any[]
    for i in 1:n_react
        push!(reactants_list, hcat([reactants[j][:,i] for j in 1:numofRun]...))
    end
    return reactants_list
end

function sub_plot(row::Int,column::Int,time_list::Vector,reactants_list::Vector;N=59)
    n_react = Int(row*column)
    dist_list = Any[];
    mean_list = Any[];
    for j in 1:n_react
        push!(dist_list,histogram(reactants_list[j][end,:],legend=false,bins=0:N,normalize=:pdf))
        push!(mean_list,plot(time_list,mean(reactants_list[j],dims=2),legend=false, ))
    end
    dist_plot = plot(dist_list...,layout = (row,column),size=(column*500,row*500))
    mean_plot = plot(mean_list...,layout = (row,column),size=(column*500,row*500))
    dist_plot, mean_plot
end

function SSA_signal(parms_mat::Array,parms::Vector,numofRun)
    data = @time pmap(iter->tissue_delay_para(parms_mat,parms,iter),1:numofRun)
    time_list = data[1][1]
    reactants = [data[i][2] for i in 1:numofRun]
    time_list, reactants
end

function plot_histo(data::Vector)
    max_np = ceil(maximum(data))
    min_np = 0
    edge = collect(min_np:1:max_np)
    H = fit(Histogram,data,edge)

    saved=zeros(length(H.weights),2);
    saved[:,1] = edge[1:end-1];
    saved[:,2] = H.weights/length(data);
    saved[:,1],saved[:,2]
end

function sum_with_non(vec::Vector)
    if length(vec)==0
        return 0
    else
        return sum(vec)
    end
end

function line_graph(n::Int64)
    if n == 1
        return Dict(1 => [1])
    end
    graph = Dict()
    graph[1] = [2]
    for i=2:n-1
        graph[i] = [i-1, i+1]
    end
    graph[n] = [n-1]
    return graph
end

function circle_graph(n::Int64)
    if n == 1
        return Dict(1 => [1])
    elseif n == 2
        return Dict(1 => [2], 2 => [1])
    end
    
    graph = Dict()
    graph[1] = [2, n]
    for i=2:n-1
        graph[i] = [i-1, i+1]
    end
    graph[n] = [n-1, 1]
    return graph
end

function grids_graph(n::Int64)
    if n == 1
        return Dict(1 => [1])
    elseif n == 4 
        return Dict(1 => [2, 3], 2 => [1, 4],
                    3 => [1, 4], 4 => [2, 3],)
    end
    @assert isinteger(sqrt(n)) "Sqrt of $n is not a integer. "
    s = sqrt(n)
    dis = [-s, s, -1, 1]
    graph = Dict()
    for i in 0:s-1
        for j in 0:s-1
            idx = Int((i * s) + j + 1)
            lb, rb = i*s + 1, (i+1)*s
            # Neighbors
            top    = idx + dis[1] < 1 ? idx + s*(s-1) : idx + dis[1]   
            bottom = idx + dis[2] > n ? idx - s*(s-1) : idx + dis[2]   
            left   = idx + dis[3] < lb ? idx + s - 1 : idx + dis[3]    
            right  = idx + dis[4] > rb ? idx - s + 1 : idx + dis[4]
            graph[idx] = Int.([top, bottom, left, right])
        end
    end
    return graph
end
