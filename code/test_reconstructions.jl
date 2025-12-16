#====================================
Algorithms of the Mind final project

Niels Verosky
=====================================#

include("sequence_memory.jl")  # generative sequence model


### define sampling strategies

"""
create a fixed sampling strategy of every k symbols
"""
function create_sample_fixed(k::Int)
    return sequence::Vector{Int} -> k
end


"""
logarithmic sampling strategy
"""
function sample_logarithmic(sequence::Vector{Int})
    return Int(floor(length(sequence) / log2(length(sequence))))
end


### test reconstruction accuracy for fixed sampling strategies

const sequence_dir = "Sequences"
num_pieces = length(readdir(sequence_dir))
scores = zeros(num_pieces)
baselines = zeros(num_pieces)
fixed_sr = 10  # sampling interval to test

for (piece, filename) in enumerate(readdir(sequence_dir))
    try
        print(string(filename, "..."))
        score = reconstruction_accuracy(joinpath(sequence_dir, filename), create_sample_fixed(fixed_sr))
        scores[piece] = score
        baseline = permuted_accuracy(joinpath(sequence_dir, filename), create_sample_fixed(fixed_sr))
        baselines[piece] = baseline
        println(string(score, " vs. ", baseline))
    catch e
        println(e)
    end
end

df = DataFrame(scores=scores, baselines=baselines)
CSV.write(string("scores_sr_", fixed_sr, ".csv"), df)


### test reconstruction accuracy by position

truncate_len = 60
num_pieces = length(readdir(sequence_dir))
scores = zeros(truncate_len)
baselines = zeros(truncate_len)
fixed_sr = 10

for (piece, filename) in enumerate(readdir(sequence_dir))
    try
        println(string(filename, "..."))
        scores .+= reconstruction_accuracy(joinpath(sequence_dir, filename), create_sample_fixed(fixed_sr), by_position=true)
        baselines .+= permuted_accuracy(joinpath(sequence_dir, filename), create_sample_fixed(fixed_sr), by_position=true)
    catch e
        println(e)
    end
end

scores /= num_pieces
baselines /= num_pieces

df = DataFrame(scores=scores, baselines=baselines)
CSV.write(string("scores_by_position_", fixed_sr, ".csv"), df)


### test reconstruction accuracy for logarithmic sampling strategy

const sequence_len_dir = "Sequence_Lengths"
seq_lens = (8, 16, 32, 48, 64, 96, 128, 144, 192, 240, 256, 288)
scores = zeros(length(seq_lens))
baselines = zeros(length(seq_lens))

for (i, seq_len) in enumerate(seq_lens)
    print(string(seq_len, "..."))
    score = 0
    baseline = 0
    num_pieces = 0
    for (piece, filename) in enumerate(readdir(sequence_len_dir))
        if occursin(string("len", seq_len), filename)
            score += reconstruction_accuracy(joinpath(sequence_len_dir, filename), sample_logarithmic)
            baseline += permuted_accuracy(joinpath(sequence_len_dir, filename), sample_logarithmic)
            num_pieces += 1
        end
    end
    score /= num_pieces
    baseline /= num_pieces
    println(string(score, " vs. ", baseline))
    scores[i] = score
    baselines[i] = baseline
end

df = DataFrame(scores=scores, baselines=baselines)
CSV.write(string("scores_var.csv"), df)