# Topology of Whole Cell 11 * 11
# +----+----+----+----+----+----+---------+----+----+----+
# |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 |
# +----+----+----+----+----+----+----+----+----+----+----+
# | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
# +----+----+----+----+----+----+----+----+----+----+----+
# | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 |
# +----+---------┌────────────────────────┐--------------+
# | 34 | 35 | 36 │ 37 | 38 | 39 | 40 | 41 │ 42 | 43 | 44 |
# +----+---------│------------------------│--------------+
# | 45 | 46 | 47 │ 48 | 49 | 50 | 51 | 52 │ 53 | 54 | 55 |
# +----+---------│---------┌────┐---------│--------------+
# | 56 | 57 | 58 │ 59 | 60 │ 61 │ 62 | 63 │ 64 | 65 | 66 |
# +----+---------│---------└────┘---------│--------------+
# | 67 | 68 | 69 │ 70 | 71 | 72 | 73 | 74 │ 75 | 76 | 77 |
# +----+---------│------------------------│--------------+
# | 78 | 79 | 80 │ 81 | 82 | 83 | 84 | 85 │ 86 | 87 | 88 |
# +----+---------└────────────────────────┘--------------+
# | 89 | 90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 |
# +----+-------------------------------------------------+
# | 100| 101| 102| 103| 104| 105| 106| 107| 108| 109| 110|
# +----+-------------------------------------------------+
# | 111| 112| 113| 114| 115| 116| 117| 118| 119| 120| 121|
# +----+----+----+----+----+----+----+----+----+----+----+
# 

# Consider four directions
dx = [-1,  1,  0,  0]
dy = [ 0,  0, -1,  1]

function reacDiffusType(i, j; type="")
    # This function returns these types of diffusion under the given parameter：
    # ["Nucleus Diffusion", "Nucleus Diffusion", "Boundary Diffusion", "non-Diffusion"]
    # i is the index of the cell from which the substance is transmitted; j is the index of the cell where the substance is received
    # Please make sure that j is the neighbor of i before passing it to the function. The function does not check if j is a neighbor of i
    # This function only checks whether i and j belong to the same medium (nucleus or cytoplasm) or satisfy the boundary penetration condition
    @assert type in ["Protein", "mRNA"] "'type' of function reacDiffusType Error: Available type: \"Protein\", \"mRNA\""
    
    if (i in Nucleus_Cells && j in Nucleus_Cells)
        return "Nucleus Diffusion"
    elseif (i in Cytoplasm_Cells && j in Cytoplasm_Cells)
        return "Cytoplasm Diffusion"
    end
    
    if type == "Protein" && ((i in Cytoplasm_Cells && j in Nucleus_Cells) || 
                             (i in Nucleus_Cells  &&  j in Cytoplasm_Cells))
        return "Boundary Diffusion"
    elseif type == "mRNA" && (i in Nucleus_Cells && j in Cytoplasm_Cells)
        return "Boundary Diffusion"
    end
    return "non-Diffusion"
end

function addNeiborCell!(D::Dict, i::Int, j::Int)
    # D is graph,i is self label,j is neighbor label
    if haskey(D, i)
        append!(D[i], j)
    else
        D[i] = [j]
    end
end

# Define size of the cell
sz = 11
n_cells = sz * sz

# Create the entire topology
topo_matrix = reshape(vec(1:sz^2), sz, sz)'  

# Define index of gene, nucleus and cytoplasm
Gene_Cell = Int((n_cells + 1) / 2) 
Nucleus_Cells = [37:41; 48:52; 59:63; 70:74; 81:85]
Cytoplasm_Cells = filter(x -> !(x in Nucleus_Cells), collect(1:n_cells)) 
Nucleus_Boundary_Cells = [37:41; 48:sz:70; 52:sz:74; 81:85]
Cytoplasm_Boundary_Cells = [26:30; 36:sz:80; 42:sz:86; 92:96; 37:41; 48:sz:70; 52:sz:74; 81:85]

# Define diffusion graph
mRNA_Nucleus_graph = Dict()
mRNA_Cytoplasm_graph = Dict()
mRNA_Boundary_graph = Dict()
Protein_Nucleus_graph = Dict()
Protein_Cytoplasm_graph = Dict()
Protein_Boundary_graph = Dict()

for i in 1:sz
    for j in 1:sz
        cell_idx = topo_matrix[i, j]
        for k in 1:4
            ni = i + dx[k]
            nj = j + dy[k]
            if ni <= sz && ni > 0 && nj <= sz && nj > 0
                neibor_cell_idx = topo_matrix[ni, nj]
                # Add diffusion type of mRNA
                mRNA_diffus_type = reacDiffusType(cell_idx, neibor_cell_idx, type="mRNA")  
                if mRNA_diffus_type == "Nucleus Diffusion"
                    addNeiborCell!(mRNA_Nucleus_graph, cell_idx, neibor_cell_idx)
                elseif mRNA_diffus_type == "Cytoplasm Diffusion"
                    addNeiborCell!(mRNA_Cytoplasm_graph, cell_idx, neibor_cell_idx)
                elseif mRNA_diffus_type == "Boundary Diffusion"
                    addNeiborCell!(mRNA_Boundary_graph, cell_idx, neibor_cell_idx)
                end
                # Add diffusion type of Protein
                Protein_diffus_type = reacDiffusType(cell_idx, neibor_cell_idx, type="Protein")  
                if Protein_diffus_type == "Nucleus Diffusion"
                    addNeiborCell!(Protein_Nucleus_graph, cell_idx, neibor_cell_idx)
                elseif Protein_diffus_type == "Cytoplasm Diffusion"
                    addNeiborCell!(Protein_Cytoplasm_graph, cell_idx, neibor_cell_idx)
                elseif Protein_diffus_type == "Boundary Diffusion"
                    addNeiborCell!(Protein_Boundary_graph, cell_idx, neibor_cell_idx)
                end

            end
        end
    end
end