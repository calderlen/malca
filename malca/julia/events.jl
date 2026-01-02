module EventsBayes

import Base: @kwdef
using DataFrames
using Statistics
using StatsBase: mad
using LogExpFunctions: logsumexp
using LsqFit
using LinearAlgebra

include("baseline.jl")
include("df_utils.jl")
include("lc_utils.jl")
using .Baseline: per_camera_gp_baseline, DEFAULT_JITTER
using .DfUtils: clean_lc, biweight_location, biweight_scale
using .LcUtils: read_lc_dat2

const MAG_BINS = ["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"]

const DEFAULT_BASELINE_KWARGS = Dict(
    :S0 => 0.0005,
    :w0 => 0.0031415926535897933,
    :q => 0.7,
    :jitter => DEFAULT_JITTER,
    :sigma_floor => nothing,
    :add_sigma_eff_col => true,
)

gaussian(t, amp, t0, sigma, baseline) = baseline .+ amp .* exp.(-0.5 .* ((t .- t0) ./ sigma) .^ 2)

function paczynski(t, amp, t0, tE, baseline)
    tE = max.(abs.(tE), 1e-5)
    baseline .+ amp ./ sqrt.(1 .+ ((t .- t0) ./ tE) .^ 2)
end

function log_gaussian(x, μ, σ)
    x = Float64.(x)
    μ = Float64.(μ)
    σ = clamp.(Float64.(σ), 1e-12, Inf)
    z = (x .- μ) ./ σ
    -0.5 .* z .^ 2 .- log.(σ) .- 0.5 .* log(2π)
end

function logit_spaced_grid(; p_min=1e-4, p_max=1 - 1e-4, n=80)
    p_min = clamp(float(p_min), 1e-12, 1 - 1e-12)
    p_max = clamp(float(p_max), 1e-12, 1 - 1e-12)
    q_min = log(p_min / (1 - p_min))
    q_max = log(p_max / (1 - p_max))
    q = range(q_min, q_max; length=n)
    1.0 ./ (1 .+ exp.(-q))
end

function default_mag_grid(baseline_mag::Float64, mags::AbstractVector, kind::String; n=60)
    mags_finite = filter(isfinite, mags)
    isempty(mags_finite) && throw(ArgumentError("No finite magnitude values for grid construction"))
    lo, hi = quantile(mags_finite, [0.05, 0.95])
    if !(isfinite(lo) && isfinite(hi))
        med = median(mags_finite)
        lo, hi = med - 0.5, med + 0.5
    end
    spread = max(hi - lo, 0.05)

    start = stop = baseline_mag
    if kind == "dip"
        start = baseline_mag + 0.02
        stop = max(baseline_mag + 0.02, hi + 0.5 * spread)
    elseif kind == "jump"
        start = min(baseline_mag - 0.02, lo - 0.5 * spread)
        stop = baseline_mag - 0.02
    else
        throw(ArgumentError("kind must be 'dip' or 'jump'"))
    end

    if start == stop
        stop = start + (kind == "dip" ? 0.1 : -0.1)
    end

    collect(range(start, stop; length=n))
end

function robust_median_dt_days(jd)
    jd = Float64.(jd)
    length(jd) < 2 && return NaN
    dt = diff(jd)
    dt = dt[isfinite.(dt) .& (dt .> 0)]
    isempty(dt) && return NaN
    median(dt)
end

function bic(resid, err, n_params)
    err = clamp.(Float64.(err), 1e-9, Inf)
    chi2 = sum((resid ./ err) .^ 2; init=0.0)
    n_points = length(resid)
    n_points == 0 && return Inf
    chi2 + n_params * log(n_points)
end

function classify_run_morphology(jd, mag, err, run_idx; kind::String="dip")
    pad = 5
    start_i = max(1, first(run_idx) - pad)
    end_i = min(length(jd), last(run_idx) + pad + 1)
    t_seg = jd[start_i:end_i]
    y_seg = mag[start_i:end_i]
    e_seg = err[start_i:end_i]

    if length(t_seg) < 4
        return Dict("morphology" => "none", "bic" => NaN, "delta_bic_null" => 0.0, "params" => Dict())
    end

    baseline_guess = median(y_seg)
    abs_diff = abs.(y_seg .- baseline_guess)
    peak_local_idx = argmax(abs_diff)

    t0_guess = t_seg[peak_local_idx]
    amp_guess = y_seg[peak_local_idx] - baseline_guess
    sigma_guess = max((t_seg[end] - t_seg[1]) / 4.0, 0.01)

    resid_null = y_seg .- baseline_guess
    bic_null = bic(resid_null, e_seg, 1)

    best_bic = bic_null
    best_model = "noise"
    best_params = Dict{String, Float64}()

    # Gaussian fit
    try
        model_g(p, t) = gaussian(t, p[1], p[2], p[3], p[4])
        p0 = [amp_guess, t0_guess, sigma_guess, baseline_guess]
        fit_g = curve_fit(model_g, t_seg, y_seg, p0; weights=1 ./ e_seg, maxIter=2000)
        popt_g = coef(fit_g)
        resid_g = y_seg .- model_g(popt_g, t_seg)
        bic_g = bic(resid_g, e_seg, 4)
        is_valid = kind == "dip" ? (popt_g[1] > 0) : (popt_g[1] < 0)
        if is_valid && bic_g < (best_bic - 10)
            best_bic = bic_g
            best_model = "gaussian"
            best_params = Dict(
                "amp" => popt_g[1],
                "t0" => popt_g[2],
                "sigma" => popt_g[3],
                "baseline" => popt_g[4],
            )
        end
    catch
    end

    if kind == "jump"
        try
            model_p(p, t) = paczynski(t, p[1], p[2], p[3], p[4])
            p0 = [-abs(amp_guess), t0_guess, sigma_guess, baseline_guess]
            fit_p = curve_fit(model_p, t_seg, y_seg, p0; weights=1 ./ e_seg, maxIter=2000)
            popt_p = coef(fit_p)
            resid_p = y_seg .- model_p(popt_p, t_seg)
            bic_p = bic(resid_p, e_seg, 4)
            is_valid_p = popt_p[1] < 0
            if is_valid_p && bic_p < (best_bic - 10)
                best_bic = bic_p
                best_model = "paczynski"
                best_params = Dict(
                    "amp" => popt_p[1],
                    "t0" => popt_p[2],
                    "tE" => popt_p[3],
                    "baseline" => popt_p[4],
                )
            end
        catch
        end
    end

    Dict(
        "morphology" => best_model,
        "bic" => float(best_bic),
        "delta_bic_null" => float(bic_null - best_bic),
        "params" => best_params,
    )
end

function build_runs(trig_idx, jd; allow_gap_points::Int=1, max_gap_days=nothing)
    jd = Float64.(jd)
    trig_idx = Int.(trig_idx[(trig_idx .>= 1) .& (trig_idx .<= length(jd))])
    isempty(trig_idx) && return Vector{Vector{Int}}()

    trig_idx = sort(unique(trig_idx))
    cad = robust_median_dt_days(jd)
    if max_gap_days === nothing
        max_gap_days = isfinite(cad) ? max(5.0 * cad, 5.0) : 5.0
    end
    max_index_step = allow_gap_points + 1

    runs = Vector{Vector{Int}}()
    cur = [trig_idx[1]]
    for k in 2:length(trig_idx)
        i_prev = cur[end]
        i = trig_idx[k]
        idx_step = i - i_prev
        dt = jd[i] - jd[i_prev]
        if (idx_step <= max_index_step) && isfinite(dt) && (dt <= max_gap_days)
            push!(cur, i)
        else
            push!(runs, copy(cur))
            cur = [i]
        end
    end
    push!(runs, cur)
    runs
end

function filter_runs(
    runs::Vector{Vector{Int}},
    jd,
    score_vec;
    min_points::Int=3,
    min_duration_days=nothing,
    per_point_threshold=nothing,
    sum_threshold=nothing,
)
    jd = Float64.(jd)
    score_vec = Float64.(score_vec)

    cad = robust_median_dt_days(jd)
    if min_duration_days === nothing
        min_duration_days = isfinite(cad) ? max(2.0 * cad, 2.0) : 2.0
    end
    min_duration_days = float(min_duration_days)

    kept = Vector{Vector{Int}}()
    summaries = Vector{Dict{Symbol, Any}}()

    for r in runs
        if isempty(r)
            continue
        end
        n = length(r)
        dur = n >= 2 ? (jd[r[end]] - jd[r[1]]) : 0.0
        vals = score_vec[r]
        run_max = any(isfinite.(vals)) ? maximum(filter(isfinite, vals)) : NaN
        run_sum = any(isfinite.(vals)) ? sum(filter(isfinite, vals)) : NaN

        ok = true
        if n < min_points
            ok = false
        end
        if dur < min_duration_days
            ok = false
        end
        if per_point_threshold !== nothing
            ok &= isfinite(run_max) && (run_max >= float(per_point_threshold))
        end
        if sum_threshold !== nothing
            ok &= isfinite(run_sum) && (run_sum >= float(sum_threshold))
        end

        push!(summaries, Dict(
            :start_idx => r[1],
            :end_idx => r[end],
            :n_points => n,
            :start_jd => jd[r[1]],
            :end_jd => jd[r[end]],
            :duration_days => dur,
            :run_max => run_max,
            :run_sum => run_sum,
            :kept => ok,
        ))

        ok && push!(kept, r)
    end

    kept, summaries
end

function summarize_kept_runs(kept_runs::Vector{Vector{Int}}, jd, score_vec)
    jd = Float64.(jd)
    score_vec = Float64.(score_vec)

    if isempty(kept_runs)
        return Dict(
            :n_runs => 0,
            :max_run_points => 0,
            :max_run_duration => NaN,
            :max_run_sum => NaN,
            :max_run_max => NaN,
        )
    end

    max_pts = 0
    max_dur = -Inf
    max_sum = -Inf
    max_max = -Inf

    for r in kept_runs
        max_pts = max(max_pts, length(r))
        if length(r) >= 2
            max_dur = max(max_dur, jd[r[end]] - jd[r[1]])
        else
            max_dur = max(max_dur, 0.0)
        end
        vals = score_vec[r]
        if any(isfinite.(vals))
            max_sum = max(max_sum, sum(filter(isfinite, vals)))
            max_max = max(max_max, maximum(filter(isfinite, vals)))
        end
    end

    Dict(
        :n_runs => max_pts == 0 ? 0 : length(kept_runs),
        :max_run_points => max_pts,
        :max_run_duration => isfinite(max_dur) ? max_dur : NaN,
        :max_run_sum => isfinite(max_sum) ? max_sum : NaN,
        :max_run_max => isfinite(max_max) ? max_max : NaN,
    )
end

function bayesian_event_significance(
    df::DataFrame;
    kind::String="dip",
    mag_col::Symbol=:mag,
    err_col::Symbol=:error,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs::Dict=DEFAULT_BASELINE_KWARGS,
    df_base::Union{Nothing, DataFrame}=nothing,
    use_sigma_eff::Bool=true,
    require_sigma_eff::Bool=false,
    p_min=nothing,
    p_max=nothing,
    p_points::Int=80,
    mag_grid::Union{Nothing, AbstractVector}=nothing,
    trigger_mode::String="logbf",
    logbf_threshold::Float64=5.0,
    significance_threshold::Float64=99.99997,
    run_min_points::Int=3,
    run_allow_gap_points::Int=1,
    run_max_gap_days=nothing,
    run_min_duration_days=nothing,
    run_sum_threshold=nothing,
    run_sum_multiplier::Float64=2.5,
    compute_event_prob::Bool=true,
)
    df = clean_lc(df)
    jd = Float64.(df[!, :JD])
    mags = Float64.(df[!, mag_col])

    mags_finite = count(isfinite, mags)
    if mags_finite == 0
        error("All magnitudes are NaN/inf after reading")
    end

    errs = err_col in names(df) ? Float64.(df[!, err_col]) : fill(0.05, length(mags))
    errs_finite = count(isfinite, errs)
    errs_positive = count(>(0), errs[isfinite.(errs)])
    if errs_finite == 0
        error("All errors are NaN/inf")
    end
    if errs_positive == 0
        error("All errors are non-positive")
    end

    used_sigma_eff = false

    if df_base === nothing && baseline_func !== nothing
        df_base = baseline_func(df; baseline_kwargs...)
    end

    baseline_mags = similar(mags)
    baseline_sources = fill("unknown", length(mags))
    if df_base === nothing
        baseline_mags .= median(filter(isfinite, mags))
        baseline_sources .= "global_median"
    else
        if :baseline in names(df_base)
            baseline_mags = Float64.(df_base[!, :baseline])
        else
            baseline_mags = Float64.(df_base[!, mag_col])
        end
        if :baseline_source in names(df_base)
            baseline_sources = string.(df_base[!, :baseline_source])
        end
        if use_sigma_eff && (:sigma_eff in names(df_base))
            errs_new = Float64.(df_base[!, :sigma_eff])
            errs_new_finite = count(isfinite, errs_new)
            errs_new_positive = count(>(0), errs_new[isfinite.(errs_new)])
            errs_new_finite == 0 && error("Baseline returned all NaN/inf sigma_eff")
            errs_new_positive == 0 && error("Baseline returned all non-positive sigma_eff")
            errs = errs_new
            used_sigma_eff = true
        elseif require_sigma_eff
            error("require_sigma_eff=true but baseline did not return sigma_eff")
        end
    end

    baseline_finite = count(isfinite, baseline_mags)
    baseline_finite == 0 && error("Baseline function returned all NaN/inf values")

    errs_finite_final = count(isfinite, errs)
    errs_positive_final = count(>(0), errs[isfinite.(errs)])
    errs_finite_final == 0 && error("All errors are NaN/inf after baseline")
    errs_positive_final == 0 && error("All errors are non-positive after baseline")

    valid_mask = isfinite.(mags) .& isfinite.(errs) .& (errs .> 0) .& isfinite.(baseline_mags)
    n_valid = count(valid_mask)
    n_valid == 0 && error("No valid points after baseline/error filtering")

    if n_valid < length(mags)
        mags = mags[valid_mask]
        errs = errs[valid_mask]
        baseline_mags = baseline_mags[valid_mask]
        baseline_sources = baseline_sources[valid_mask]
        jd = jd[valid_mask]
    end

    baseline_mag = median(filter(isfinite, baseline_mags))

    if p_min === nothing && p_max === nothing
        if kind == "dip"
            p_min, p_max = 0.5, 1 - 1e-4
        elseif kind == "jump"
            p_min, p_max = 1e-4, 0.5
        else
            error("kind must be 'dip' or 'jump'")
        end
    end

    p_grid = logit_spaced_grid(p_min=p_min, p_max=p_max, n=p_points)
    mag_grid = mag_grid === nothing ? default_mag_grid(baseline_mag, mags, kind; n=60) : Float64.(mag_grid)

    M = length(mag_grid)
    N = length(mags)

    if kind == "dip"
        log_Pb_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pb_grid = repeat(log_Pb_vec', M, 1)
        log_Pf_grid = log_gaussian(mags', mag_grid, errs')
        event_component = "faint"
    elseif kind == "jump"
        log_Pb_grid = log_gaussian(mags', mag_grid, errs')
        log_Pf_vec = log_gaussian(mags, baseline_mags, errs)
        log_Pf_grid = repeat(log_Pf_vec', M, 1)
        event_component = "bright"
        !any(isfinite.(log_Pf_vec)) && error("All baseline likelihood values are NaN/inf")
        !any(isfinite.(log_Pb_grid)) && error("All event likelihood values are NaN/inf")
    else
        error("kind must be 'dip' or 'jump'")
    end

    valid_points = mapslices(any, isfinite.(log_Pb_grid); dims=1)[:] .| mapslices(any, isfinite.(log_Pf_grid); dims=1)[:]
    n_valid_points = count(valid_points)
    total_points = size(log_Pb_grid, 2)
    n_valid_points == 0 && error("No valid likelihood contributions after baseline")

    if n_valid_points < total_points
        mags = mags[valid_points]
        errs = errs[valid_points]
        baseline_mags = baseline_mags[valid_points]
        baseline_sources = baseline_sources[valid_points]
        jd = jd[valid_points]
        log_Pb_grid = log_Pb_grid[:, valid_points]
        log_Pf_grid = log_Pf_grid[:, valid_points]
        if kind == "dip"
            log_Pb_vec = log_Pb_vec[valid_points]
        else
            log_Pf_vec = log_Pf_vec[valid_points]
        end
        N = n_valid_points
    end

    if kind == "dip"
        loglik_baseline_only = sum(log_Pb_vec)
        log_px_baseline = log_Pb_vec
        log_px_event = vec(logsumexp(log_Pf_grid; dims=1)) .- log(M)
    else
        loglik_baseline_only = sum(log_Pf_vec)
        log_px_baseline = log_Pf_vec
        log_px_event = vec(logsumexp(log_Pb_grid; dims=1)) .- log(M)
    end

    log_bf_local = log_px_event .- log_px_baseline
    max_log_bf_local = any(isfinite.(log_bf_local)) ? maximum(filter(isfinite, log_bf_local)) : NaN

    log_p = log.(p_grid)
    log_1mp = log1p.(-p_grid)

    log_Pb_weighted = reshape(log_Pb_grid, M, 1, N) .+ reshape(log_p, 1, length(log_p), 1)
    log_Pf_weighted = reshape(log_Pf_grid, M, 1, N) .+ reshape(log_1mp, 1, length(log_1mp), 1)
    log_Pb_weighted .= ifelse.(isfinite.(log_Pb_weighted), log_Pb_weighted, -Inf)
    log_Pf_weighted .= ifelse.(isfinite.(log_Pf_weighted), log_Pf_weighted, -Inf)

    log_mix = log.(exp.(log_Pb_weighted) .+ exp.(log_Pf_weighted))
    log_mix_finite = count(isfinite, log_mix)
    log_mix_finite == 0 && error("All log_mix values are NaN/inf")

    loglik = sum(log_mix; dims=3)
    loglik_finite = count(isfinite, loglik)
    loglik_total = length(loglik)
    if loglik_finite == 0
        error("All loglik values are NaN/inf before normalization")
    end

    loglik_sum = logsumexp(vec(loglik))
    isfinite(loglik_sum) || error("logsumexp(loglik) is NaN/inf")
    log_post_norm = loglik .- loglik_sum
    log_post_finite = count(isfinite, log_post_norm)
    log_post_finite == 0 && error("All log_posterior values are NaN/inf after normalization")

    best_idx = argmax(vec(log_post_norm))
    best_m_idx = rem(best_idx - 1, M) + 1
    best_p_idx = div(best_idx - 1, M) + 1
    best_mag_event = mag_grid[best_m_idx]
    best_p = p_grid[best_p_idx]

    K = length(loglik)
    log_evidence_mixture = logsumexp(vec(loglik)) - log(K)
    bayes_factor = log_evidence_mixture - loglik_baseline_only

    event_prob = compute_event_prob ? zeros(Float64, N) : nothing
    if compute_event_prob
        for j in 1:N
            loglik_excl = loglik .- log_mix[:, :, j]
            bright_num = loglik_excl .+ log_p' .+ reshape(log_Pb_grid[:, j], M, 1, 1)
            faint_num = loglik_excl .+ log_1mp' .+ reshape(log_Pf_grid[:, j], M, 1, 1)
            log_bright = logsumexp(vec(bright_num))
            log_faint = logsumexp(vec(faint_num))
            log_norm = logsumexp([log_bright, log_faint])
            bright_prob = exp(log_bright - log_norm)
            faint_prob = exp(log_faint - log_norm)
            if event_component == "faint"
                event_prob[j] = faint_prob
            else
                event_prob[j] = bright_prob
            end
        end
    end

    raw_idx = Int[]
    trigger_threshold_used = NaN
    trigger_value_max = NaN
    run_sum_threshold_eff = run_sum_threshold

    if trigger_mode == "logbf"
        per_point_thr = logbf_threshold
        score_vec = log_bf_local
        raw_idx = findall(i -> isfinite(score_vec[i]) && score_vec[i] >= per_point_thr, 1:length(score_vec))
        trigger_threshold_used = per_point_thr
        trigger_value_max = max_log_bf_local
        if run_sum_threshold === nothing
            run_sum_threshold_eff = run_sum_multiplier * per_point_thr
        end
    elseif trigger_mode == "posterior_prob"
        compute_event_prob || error("trigger_mode=posterior_prob requires compute_event_prob=true")
        thr_prob = significance_threshold > 1 ? significance_threshold / 100.0 : significance_threshold
        score_vec = event_prob
        raw_idx = findall(i -> isfinite(score_vec[i]) && score_vec[i] >= thr_prob, 1:length(score_vec))
        trigger_threshold_used = thr_prob
        trigger_value_max = maximum(score_vec)
        if run_sum_threshold === nothing
            run_sum_threshold_eff = run_min_points * thr_prob
        end
    else
        error("trigger_mode must be 'logbf' or 'posterior_prob'")
    end

    kept_runs = Vector{Vector{Int}}()
    run_summaries = Vector{Dict{Symbol, Any}}()
    event_indices = Int[]
    significant = false
    run_stats = Dict{Symbol, Any}()

    if isempty(raw_idx)
        run_stats = summarize_kept_runs(Vector{Vector{Int}}(), jd, score_vec)
    else
        runs = build_runs(raw_idx, jd; allow_gap_points=run_allow_gap_points, max_gap_days=run_max_gap_days)
        kept_runs, initial_summaries = filter_runs(
            runs,
            jd,
            score_vec;
            min_points=run_min_points,
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            sum_threshold=run_sum_threshold_eff,
        )

        for (i, r) in enumerate(kept_runs)
            summary = initial_summaries[i]
            morph_res = classify_run_morphology(jd, mags, errs, r; kind=kind)
            merge!(summary, Dict(Symbol(k) => v for (k, v) in morph_res))
            push!(run_summaries, summary)
        end

        if !isempty(kept_runs)
            event_indices = unique(vcat(kept_runs...))
            significant = true
        end
        run_stats = summarize_kept_runs(kept_runs, jd, score_vec)
    end

    Dict(
        "kind" => string(kind),
        "baseline_mag" => float(baseline_mag),
        "best_mag_event" => float(best_mag_event),
        "best_p" => float(best_p),
        "log_bf_local" => log_bf_local,
        "max_log_bf_local" => float(isfinite(max_log_bf_local) ? max_log_bf_local : NaN),
        "event_probability" => event_prob,
        "used_sigma_eff" => used_sigma_eff,
        "trigger_mode" => string(trigger_mode),
        "trigger_threshold" => float(trigger_threshold_used),
        "trigger_max" => float(isfinite(trigger_value_max) ? trigger_value_max : NaN),
        "event_indices" => event_indices,
        "significant" => significant,
        "run_sum_threshold" => float(run_sum_threshold_eff),
        "run_summaries" => run_summaries,
        :n_runs => get(run_stats, :n_runs, 0),
        :max_run_points => get(run_stats, :max_run_points, 0),
        :max_run_duration => get(run_stats, :max_run_duration, NaN),
        :max_run_sum => get(run_stats, :max_run_sum, NaN),
        :max_run_max => get(run_stats, :max_run_max, NaN),
        "bayes_factor" => float(bayes_factor),
        "log_evidence_mixture" => float(log_evidence_mixture),
        "log_evidence_baseline" => float(loglik_baseline_only),
        "baseline_source" => isempty(baseline_sources) ? "unknown" : join(sort(unique(string.(baseline_sources))), ","),
        "p_grid" => p_grid,
        "mag_grid" => mag_grid,
    )
end

function run_bayesian_significance(
    df::DataFrame;
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs::Dict=DEFAULT_BASELINE_KWARGS,
    p_points::Int=80,
    p_min_dip=nothing,
    p_max_dip=nothing,
    p_min_jump=nothing,
    p_max_jump=nothing,
    mag_grid_dip=nothing,
    mag_grid_jump=nothing,
    trigger_mode::String="logbf",
    logbf_threshold_dip::Float64=5.0,
    logbf_threshold_jump::Float64=5.0,
    significance_threshold::Float64=99.99997,
    run_min_points::Int=3,
    run_allow_gap_points::Int=1,
    run_max_gap_days=nothing,
    run_min_duration_days=nothing,
    run_sum_threshold=nothing,
    run_sum_multiplier::Float64=2.5,
    use_sigma_eff::Bool=true,
    require_sigma_eff::Bool=true,
    compute_event_prob::Bool=true,
)
    df = clean_lc(df)
    df_base = baseline_func !== nothing ? baseline_func(df; baseline_kwargs...) : nothing

    dip = bayesian_event_significance(
        df;
        kind="dip",
        baseline_func=nothing,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_dip,
        p_max=p_max_dip,
        p_points=p_points,
        mag_grid=mag_grid_dip,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_dip,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    jump = bayesian_event_significance(
        df;
        kind="jump",
        baseline_func=nothing,
        baseline_kwargs=baseline_kwargs,
        df_base=df_base,
        use_sigma_eff=use_sigma_eff,
        require_sigma_eff=require_sigma_eff,
        p_min=p_min_jump,
        p_max=p_max_jump,
        p_points=p_points,
        mag_grid=mag_grid_jump,
        trigger_mode=trigger_mode,
        logbf_threshold=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        run_min_points=run_min_points,
        run_allow_gap_points=run_allow_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,
        run_sum_threshold=run_sum_threshold,
        run_sum_multiplier=run_sum_multiplier,
        compute_event_prob=compute_event_prob,
    )

    Dict(:dip => dip, :jump => jump)
end

end # module
