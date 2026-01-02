module DfUtils

using DataFrames
using Statistics
using StatsBase: mad

"""
    clean_lc(df)

Filter light curve rows to finite JD/mag, optional saturated==0 and reasonable errors,
then sort by JD.
"""
function clean_lc(df::DataFrame)
    mask = trues(nrow(df))
    if :saturated in names(df)
        mask .&= df.saturated .== 0
    end
    mask .&= .!ismissing.(df.JD) .& .!ismissing.(df.mag)
    if :error in names(df)
        mask .&= .!ismissing.(df.error) .& (df.error .> 0) .& (df.error .< 1)
    end
    out = df[mask, :]
    sort!(out, :JD)
    out
end

year_to_jd(year) = (year - 1995) * 365.25 + (2449718.5 - 2450000.0)
jd_to_year(jd) = 1995 + ((jd + 2450000.0) - 2449718.5) / 365.25

# Biweight helpers (astropy-style)
function biweight_location(x; c::Float64=6.0, eps::Float64=1e-24)
    xf = filter(isfinite, x)
    isempty(xf) && return NaN
    M = median(xf)
    s = mad(xf; center=M, normalize=true)
    s = s <= eps ? eps : s
    u = (xf .- M) ./ (c * s)
    keep = abs.(u) .< 1
    y = xf[keep]
    u = u[keep]
    isempty(y) && return M
    w = (1 .- u .^ 2) .^ 2
    denom = sum(w)
    denom == 0 && return M
    M + sum((y .- M) .* w) / denom
end

function biweight_scale(x; c::Float64=6.0, eps::Float64=1e-24)
    xf = filter(isfinite, x)
    isempty(xf) && return 0.0
    M = median(xf)
    s0 = mad(xf; center=M, normalize=true)
    s0 = s0 <= eps ? eps : s0
    u = (xf .- M) ./ (c * s0)
    keep = abs.(u) .< 1
    y = xf[keep]
    u = u[keep]
    isempty(y) && return 0.0
    term1 = (y .- M) .^ 2 .* (1 .- u .^ 2) .^ 4
    term2 = (1 .- u .^ 2) .* (1 .- 5 .* u .^ 2)
    num = sqrt(length(y)) * sqrt(sum(term1))
    den = abs(sum(term2))
    den == 0 && return 0.0
    num / den
end

# Simplified peak finding with optional prominence/height/distance filters
function _find_peaks(values::AbstractVector{<:Real};
    prominence::Float64=0.0,
    distance::Int=1,
    height::Float64=-Inf,
)
    n = length(values)
    n < 3 && return Int[]
    candidates = Int[]
    for i in 2:(n - 1)
        v = values[i]
        (v > values[i - 1]) && (v > values[i + 1]) || continue
        v < height && continue
        push!(candidates, i)
    end
    if prominence > 0 && !isempty(candidates)
        keep = Int[]
        for i in candidates
            left_min = minimum(values[1:i])
            right_min = minimum(values[i:end])
            prom = values[i] - max(left_min, right_min)
            prom >= prominence && push!(keep, i)
        end
        candidates = keep
    end
    if distance > 1 && !isempty(candidates)
        # Greedy keep highest peaks first
        order = sort(candidates; by=i -> -values[i])
        kept = Int[]
        for i in order
            if all(abs(i - k) >= distance for k in kept)
                push!(kept, i)
            end
        end
        candidates = sort(kept)
    end
    candidates
end

function peak_search_residual_baseline(
    df::DataFrame;
    prominence::Float64=0.17,
    distance::Int=25,
    height::Float64=0.3,
    width::Int=2, # unused placeholder for parity
    apply_box_filter::Bool=true,
    max_dips::Int=10,
    max_std::Float64=0.15,
    max_peaks_per_time::Float64=0.015,
)
    mag = Float64.(df[!, :mag])
    jd = Float64.(df[!, :JD])
    finite_mag = filter(isfinite, mag)
    meanmag = !isempty(finite_mag) ? mean(finite_mag) : NaN

    values = if :resid in names(df)
        v = Float64.(df[!, :resid])
        map(x -> isfinite(x) ? x : 0.0, v)
    else
        mag .- meanmag
    end

    peak_idx = _find_peaks(values; prominence, distance, height)
    n_peaks = length(peak_idx)

    if apply_box_filter
        jd_span = length(jd) > 1 ? (jd[end] - jd[1]) : 0.0
        peaks_per_time = jd_span > 0 ? n_peaks / jd_span : Inf
        std_mag = std(filter(isfinite, values))

        if (n_peaks == 0) || (n_peaks >= max_dips) || (peaks_per_time > max_peaks_per_time) || (std_mag > max_std)
            return Int[], meanmag, 0
        end
    end

    return peak_idx, meanmag, n_peaks
end

function peak_search_biweight_delta(
    df::DataFrame;
    sigma_threshold::Float64=3.0,
    distance::Int=25,
    width::Int=1, # unused placeholder for parity
    prominence::Float64=0.0,
    apply_box_filter::Bool=false,
    max_dips::Int=10,
    max_peaks_per_time::Float64=0.015,
    max_std_sigma::Float64=2.5,
    mag_col::Symbol=:mag,
    t_col::Symbol=:JD,
    err_col::Symbol=:error,
    biweight_c::Float64=6.0,
    eps::Float64=1e-6,
)
    mag = mag_col in names(df) ? Float64.(df[!, mag_col]) : Float64[]
    jd = t_col in names(df) ? Float64.(df[!, t_col]) : Float64[]
    err = err_col in names(df) ? Float64.(df[!, err_col]) : fill(NaN, length(mag))

    finite_m = isfinite.(mag)
    if any(finite_m)
        R = biweight_location(mag[finite_m]; c=biweight_c)
        S = biweight_scale(mag[finite_m]; c=biweight_c)
    else
        R = NaN
        S = 0.0
    end
    if !isfinite(S) || S < 0
        S = 0.0
    end

    err2 = map(x -> isfinite(x) ? x^2 : 0.0, err)
    denom = sqrt.(err2 .+ S^2)
    denom = map(d -> d > 0 ? d : eps, denom)

    delta = (mag .- R) ./ denom
    values = map(x -> (isfinite(x) ? x : 0.0), delta)

    peak_idx = _find_peaks(values; prominence, distance, height=sigma_threshold)
    n_peaks = length(peak_idx)

    if apply_box_filter
        jd_span = length(jd) > 1 ? (jd[end] - jd[1]) : 0.0
        peaks_per_time = jd_span > 0 ? n_peaks / jd_span : Inf
        std_sig = std(filter(isfinite, values))

        if (n_peaks == 0) || (n_peaks >= max_dips) || (peaks_per_time > max_peaks_per_time) || (std_sig > max_std_sigma)
            return Int[], R, 0
        end
    end

    return peak_idx, R, n_peaks
end

"""
    empty_metrics(prefix)

Return a Dict of empty metrics with keys prefixed.
"""
function empty_metrics(prefix::AbstractString)
    vals = Dict(
        "n_dip_runs" => 0,
        "n_jump_runs" => 0,
        "n_dip_points" => 0,
        "n_jump_points" => 0,
        "most_recent_dip" => NaN,
        "most_recent_jump" => NaN,
        "max_depth" => NaN,
        "max_height" => NaN,
        "max_dip_duration" => NaN,
        "max_jump_duration" => NaN,
        "dip_fraction" => NaN,
        "jump_fraction" => NaN,
    )
    out = Dict{String, Any}()
    for (k, v) in vals
        out["$(prefix)_$(k)"] = v
    end
    out["$(prefix)_is_dip_dominated"] = false
    out
end

end # module
