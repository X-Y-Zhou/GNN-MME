using DelimitedFiles
using Plots
default(lw=3, label=false)

data_table = readdlm("Figure2_d-f/results/computing_time.csv", ',')

SSA_2_v1  = float.(data_table[2, 2:7])
SSA_2_v2  = float.(data_table[3, 2:7])
SSA_10_v1 = float.(data_table[4, 2:7])
SSA_10_v2 = float.(data_table[5, 2:7])

NN_10_v1 = float.(data_table[6, 2:7]) + SSA_2_v1
NN_10_v2 = float.(data_table[7, 2:7])  # The calculation time of SSA is not considered in the prediction

scale_list = [2, 2.5, 3, 3.5, 4, 4.5]
sample_size_list = @. Int(round(10 ^ scale_list))
fig_colors = reshape(palette(:tab10)[:], 1, :);
f1 = plot(title="Topology extrapolation", xscale=:log10, xlims=(10^1.5, 10^5.0), ylims=(10^0, 10^4.), yscale=:log10, ylabel="Time (s)", legend=:topleft);
f1 = plot!(sample_size_list, SSA_10_v1, label="SSA (10 cells)", c=fig_colors[1], msw=2, msc=fig_colors[1])
f1 = plot!(sample_size_list, SSA_10_v1, st=:scatter, c=fig_colors[1])
f1 = plot!(sample_size_list, NN_10_v1, label="NN (10 cells)", c=fig_colors[2], msw=2, msc=fig_colors[2])
f1 = plot!(sample_size_list, NN_10_v1, st=:scatter, msw=0., c=fig_colors[2])

f2 = plot(title="Topology+kinetic extrapolation", xscale=:log10, xlims=(10^1.5, 10^5.0), ylims=(10^0, 10^4.), yscale=:log10, yticks=false, legend=:topleft);
f2 = plot!(sample_size_list, SSA_10_v2, label="SSA (10 cells)", c=fig_colors[1], msw=2, msc=fig_colors[1])
f2 = plot!(sample_size_list, SSA_10_v2, st=:scatter, msw=0., c=fig_colors[1])
f2 = plot!(sample_size_list, NN_10_v2, label="NN (10 cells)", c=fig_colors[2], msw=2, msc=fig_colors[2])
f2 = plot!(sample_size_list, NN_10_v2, st=:scatter, msw=0., c=fig_colors[2])
plot(f1, f2, layout=grid(1, 2), size=(700, 250))
#savefig("Figure2_d-f/results/Figure2_e.pdf")
 