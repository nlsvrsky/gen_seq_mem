#====================================
Algorithms of the Mind final project

Niels Verosky
=====================================#

using Gen
using CSV
using DataFrames
using Random


# fixed parameters
const n = 12  # number of symbols (notes)
const λ_short = .5  # faster-decaying temporal discounting constant
const λ_long = .9  # slower-decaying temporal discounting constant
const λ_unroll = λ_long  # proposal temporal discounting constant
const σ = .01  # working memory noise 


"""
generate a working memory snapshot
"""
@gen function snapshot(k::Int, prev_subsequence::Union{Array{Int}, Nothing}, sr::Int)
    # initialize temporal discounting weights
    λ_short_w = λ_short .^ ((sr-1):-1:0) 
    λ_long_w = λ_long .^ ((sr-1):-1:0) 
    
    # first sample a sequence
    subsequence = zeros(Int, n, sr)
    for t in 1:sr
        scale_degree = {:subsequence => t} ~ uniform_discrete(1, n)
        subsequence[scale_degree, t] = 1  
    end 

    # then generate a memory snapshot
    mu_short = subsequence * λ_short_w
    mu_long = subsequence * λ_long_w

    snapshot_short ~ broadcasted_normal(mu_short, σ)
    snapshot_long ~ broadcasted_normal(mu_long, σ)

    return subsequence
end


"""
unfold a chain of snapshots
"""
@gen function snapshots(k_max::Int, sr::Int)
    sequence ~ Unfold(snapshot)(k_max, nothing, sr)
end


"""
custom proposal that unrolls a snapshot backward in time
"""
@gen function unroll_snapshot(t_old::Union{Trace, Nothing}, k::Int, snapshot::Vector{Float64}, sr::Int) 
    unrolling = copy(snapshot) 
    
    # unroll the current snapshot backwards
    for t in sr:-1:1 
        probs = minimum(unrolling) < 0 ? unrolling .- minimum(unrolling) : unrolling
        probs ./= sum(probs)

        scale_degree = {:sequence => k => :subsequence => t} ~ categorical(probs) 
        unrolling[scale_degree] -= λ_unroll^(sr-t)
    end 
end


"""
particle filter to infer a sequence from snapshots
"""
function memory_filter(num_particles::Int, obs_snapshots::Vector{ChoiceMap}, sr::Int, num_samples::Int) 
    # create a particle filter 
    state = initialize_particle_filter(snapshots, (1, sr), obs_snapshots[1], 
        unroll_snapshot, (nothing, 1, obs_snapshots[1][:sequence => 1 => :snapshot_long], sr), num_particles)

    # each step of the particle filter is a working memory snapshot
    for k in 2:length(obs_snapshots)
        maybe_resample!(state, ess_threshold=num_particles/2, verbose=false)
        particle_filter_step!(state, (k, sr), (UnknownChange(),), obs_snapshots[k], 
            unroll_snapshot, (k, obs_snapshots[k][:sequence => k => :snapshot_long], sr))
    end

    return sample_unweighted_traces(state, num_samples)
end


"""
observe a chain of snapshots for a given sequence
"""
function observe_sequence(sequence::Vector{Int}, sr::Int)
    k_max = Int(floor(length(sequence) / sr))
    obs = Vector{ChoiceMap}(undef, k_max)
    prev_subsequence = nothing

    # loop through snapshots
    for k in 1:k_max
        constraints = choicemap()

        # put the current subsequence into a choice map
        for t in 1:sr
            constraints[:subsequence => t] = sequence[(k-1)*sr + t]
        end

        # generate snapshots for the current subsequence
        trace, _ = generate(snapshot, (k, prev_subsequence, sr), constraints)
        choices = get_choices(trace)
        obs[k] = choicemap((:sequence => k => :snapshot_short, choices[:snapshot_short]),
                            (:sequence => k => :snapshot_long, choices[:snapshot_long]))

        # save the snapshots as observed variables
        prev_subsequence = zeros(Int, n, sr)
        for t in 1:sr
            prev_subsequence[constraints[:subsequence => t], t] = 1  
        end 
    end

    return obs
end


"""
read a sequence from a file
"""
function read_sequence(filename::String)
    df = CSV.read(filename, DataFrame; header=false)
    return df[:, 1] .+ 1  # convert 0-indexed to 1-indexed
end


"""
reconstruction accuracy for a test sequence

either over the entire sequence (by_position=false)
or over each position (by_position=true)
"""
function reconstruction_accuracy(filename::String, sampling_strategy::Function; by_position::Bool=false, truncate_len::Int=60)
    sequence = by_position ? read_sequence(filename)[1:truncate_len] : read_sequence(filename)
    
    sr = sampling_strategy(sequence)  # set sampling interval
    obs = observe_sequence(sequence, sr)  # get sequence observation 

    # perform inference
    num_samples = 100
    traces = memory_filter(1000, obs, sr, num_samples);

    # average reconstruction accuracy over the posterior samples
    total_matching = by_position ? zeros(truncate_len) : 0
    
    for trace in traces
        choices = get_choices(trace);
        post_sequence = [choices[:sequence => k => :subsequence => t] for k in 1:Int(floor(length(sequence) / sr)) for t in 1:sr]
        k_max = Int(floor(length(sequence) / sr))
        sequence = sequence[1:k_max*sr]
        post_sequence = post_sequence[1:k_max*sr]
        total_matching += by_position ? Int.(post_sequence .== sequence) : sum(post_sequence .== sequence) / length(sequence)
    end

    return total_matching / num_samples
end


"""
baseline permuted accuracy for a test sequence
"""
function permuted_accuracy(filename::String, sampling_strategy::Function; by_position::Bool=false, truncate_len::Int=60)
    sequence = by_position ? read_sequence(filename)[1:truncate_len] : read_sequence(filename)
    sr = sampling_strategy(sequence)  # set sampling interval
    k_max = Int(floor(length(sequence) / sr))
    sequence = sequence[1:k_max*sr]

    num_samples = 100

    total_matching = by_position ? zeros(truncate_len) : 0

    for _ in 1:num_samples
        permuted_sequence = shuffle(sequence)[1:k_max*sr]
        total_matching += by_position ? Int.(permuted_sequence .== sequence) : sum(permuted_sequence .== sequence) / length(sequence)
    end

    return total_matching / num_samples
end