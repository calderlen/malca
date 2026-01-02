module Baseline

using DataFrames
using Statistics
using StatsBase: mad
using LinearAlgebra

const DEFAULT_JITTER = 0.006

function global_mean_baseline(df::DataFrame; t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    m = Float64.(out[!, mag_col])
    e = Float64.(out[!, err_col])

    baseline = fill(NaN, length(m))
    resid = fill(NaN, length(m))

    good = isfinite.(m)
    if any(good)
        mean_mag = mean(m[good])
        baseline .= mean_mag
        resid .= m .- mean_mag
    end

    resid_good = isfinite.(resid)
    med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
    mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

    e_good = isfinite.(e)
    e_med = any(e_good) ? median(e[e_good]) : NaN

    mad_num = isfinite(mad_resid) ? mad_resid : 0.0
    e_med_num = isfinite(e_med) ? e_med : 0.0
    robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

    sigma_resid = resid ./ robust_std

    out[!, :baseline] .= baseline
    out[!, :resid] .= resid
    out[!, :sigma_resid] .= sigma_resid
    out
end

function global_median_baseline(df::DataFrame; t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    m = Float64.(out[!, mag_col])
    e = Float64.(out[!, err_col])

    baseline = fill(NaN, length(m))
    resid = fill(NaN, length(m))

    good = isfinite.(m)
    if any(good)
        median_mag = median(m[good])
        baseline .= median_mag
        resid .= m .- median_mag
    end

    resid_good = isfinite.(resid)
    med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
    mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

    e_good = isfinite.(e)
    e_med = any(e_good) ? median(e[e_good]) : NaN

    mad_num = isfinite(mad_resid) ? mad_resid : 0.0
    e_med_num = isfinite(e_med) ? e_med : 0.0
    robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

    sigma_resid = resid ./ robust_std

    out[!, :baseline] .= baseline
    out[!, :resid] .= resid
    out[!, :sigma_resid] .= sigma_resid
    out
end

function rolling_time_median(jd, mag; days::Float64=300.0, min_points::Int=10, min_days::Float64=30.0, past_only::Bool=true)
    n = length(jd)
    out = fill(NaN, n)
    jd_f = Float64.(jd)
    mag_f = Float64.(mag)

    for i in 1:n
        t0 = jd_f[i]
        window = days
        while window >= min_days
            lo_val, hi_val = if past_only
                t0 - window, t0
            else
                half = window / 2.0
                t0 - half, t0 + half
            end
            idx_start = searchsortedfirst(jd_f, lo_val)
            idx_end = searchsortedlast(jd_f, hi_val)
            vals = mag_f[idx_start:idx_end]
            finite_vals = vals[isfinite.(vals)]
            if length(finite_vals) >= min_points
                out[i] = median(finite_vals)
                break
            end
            window /= 2.0
        end
    end
    out
end

function rolling_time_mad(jd, resid; days::Float64=200.0, min_points::Int=10, min_days::Float64=20.0, past_only::Bool=true, add_err=nothing)
    n = length(jd)
    out = fill(NaN, n)
    jd_f = Float64.(jd)
    resid_f = Float64.(resid)

    err_is_array = add_err !== nothing && ndims(add_err) > 0
    err_array = err_is_array ? Float64.(add_err) : nothing
    err_scalar = add_err === nothing ? 0.0 : (err_is_array ? 0.0 : float(add_err))

    for i in 1:n
        t0 = jd_f[i]
        window = days
        while window >= min_days
            lo_val, hi_val = if past_only
                t0 - window, t0
            else
                half = window / 2.0
                t0 - half, t0 + half
            end
            idx_start = searchsortedfirst(jd_f, lo_val)
            idx_end = searchsortedlast(jd_f, hi_val)
            vals = resid_f[idx_start:idx_end]
            finite_vals = vals[isfinite.(vals)]
            if length(finite_vals) >= min_points
                med = median(finite_vals)
                mad_val = mad(finite_vals; center=med, normalize=true)
                if add_err !== nothing
                    err_here = err_is_array ? err_array[i] : err_scalar
                    mad_val = sqrt(mad_val^2 + err_here^2)
                end
                out[i] = max(mad_val, 1e-6)
                break
            end
            window /= 2.0
        end
    end
    out
end

function global_rolling_median_baseline(df::DataFrame; days::Float64=1000.0, min_points::Int=10, t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    t = Float64.(out[!, t_col])
    m = Float64.(out[!, mag_col])
    e = Float64.(out[!, err_col])

    base = rolling_time_median(t, m; days=days, min_points=min_points)
    resid = m .- base

    resid_good = isfinite.(resid)
    med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
    mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

    e_good = isfinite.(e)
    e_med = any(e_good) ? median(e[e_good]) : NaN

    mad_num = isfinite(mad_resid) ? mad_resid : 0.0
    e_med_num = isfinite(e_med) ? e_med : 0.0
    robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

    sigma_resid = resid ./ robust_std

    out[!, :baseline] .= base
    out[!, :resid] .= resid
    out[!, :sigma_resid] .= sigma_resid
    out
end

function global_rolling_mean_baseline(df::DataFrame; days::Float64=1000.0, min_points::Int=10, t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    t = Float64.(out[!, t_col])
    m = Float64.(out[!, mag_col])
    e = Float64.(out[!, err_col])

    baseline = fill(NaN, length(m))
    order = sortperm(t)
    t_sorted = t[order]
    m_sorted = m[order]

    for (idx_sorted, i) in enumerate(order)
        t0 = t_sorted[idx_sorted]
        window = days
        while window >= min_points
            lo = t0 - window
            hi = t0
            start = searchsortedfirst(t_sorted, lo)
            ending = searchsortedlast(t_sorted, hi)
            vals = m_sorted[start:ending]
            finite = vals[isfinite.(vals)]
            if length(finite) >= min_points
                baseline[i] = mean(finite)
                break
            end
            window /= 2.0
        end
    end

    resid = m .- baseline

    resid_good = isfinite.(resid)
    med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
    mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

    e_good = isfinite.(e)
    e_med = any(e_good) ? median(e[e_good]) : NaN

    mad_num = isfinite(mad_resid) ? mad_resid : 0.0
    e_med_num = isfinite(e_med) ? e_med : 0.0
    robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

    sigma_resid = resid ./ robust_std

    out[!, :baseline] .= baseline
    out[!, :resid] .= resid
    out[!, :sigma_resid] .= sigma_resid
    out
end

function per_camera_mean_baseline(df::DataFrame; t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error, cam_col::Symbol=Symbol("camera#"))
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    groups = cam_col in names(out) ? groupby(out, cam_col; sort=true) : [out]
    for sub in groups
        idx = parentindices(sub)[1]
        m = Float64.(sub[!, mag_col])
        e = Float64.(sub[!, err_col])

        baseline = fill(NaN, length(idx))
        resid = fill(NaN, length(idx))

        good = isfinite.(m)
        if any(good)
            cam_mean = mean(m[good])
            baseline .= cam_mean
            resid .= m .- cam_mean
        end

        resid_good = isfinite.(resid)
        med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
        mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

        e_good = isfinite.(e)
        e_med = any(e_good) ? median(e[e_good]) : NaN

        mad_num = isfinite(mad_resid) ? mad_resid : 0.0
        e_med_num = isfinite(e_med) ? e_med : 0.0
        robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

        sigma_resid = resid ./ robust_std

        out[idx, :baseline] .= baseline
        out[idx, :resid] .= resid
        out[idx, :sigma_resid] .= sigma_resid
    end
    out
end

function per_camera_median_baseline(df::DataFrame; days::Float64=300.0, min_points::Int=10, t_col::Symbol=:JD, mag_col::Symbol=:mag, err_col::Symbol=:error, cam_col::Symbol=Symbol("camera#"))
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    groups = cam_col in names(out) ? groupby(out, cam_col; sort=true) : [out]
    for sub in groups
        idx = parentindices(sub)[1]
        t = Float64.(sub[!, t_col])
        m = Float64.(sub[!, mag_col])
        e = Float64.(sub[!, err_col])

        base = rolling_time_median(t, m; days=days, min_points=min_points)
        resid = m .- base

        resid_good = isfinite.(resid)
        med_resid = any(resid_good) ? median(resid[resid_good]) : NaN
        mad_resid = any(resid_good) ? mad(resid[resid_good]; center=med_resid, normalize=true) : NaN

        e_good = isfinite.(e)
        e_med = any(e_good) ? median(e[e_good]) : NaN

        mad_num = isfinite(mad_resid) ? mad_resid : 0.0
        e_med_num = isfinite(e_med) ? e_med : 0.0
        robust_std = max(sqrt(mad_num^2 + e_med_num^2), 1e-6)

        sigma_resid = resid ./ robust_std

        out[idx, :baseline] .= base
        out[idx, :resid] .= resid
        out[idx, :sigma_resid] .= sigma_resid
    end
    out
end

function per_camera_trend_baseline(
    df::DataFrame;
    days_short::Float64=50.0,
    days_long::Float64=800.0,
    min_points::Int=10,
    last_window_guard::Float64=120.0,
    t_col::Symbol=:JD,
    mag_col::Symbol=:mag,
    err_col::Symbol=:error,
    cam_col::Symbol=Symbol("camera#"),
)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    groups = cam_col in names(out) ? groupby(out, cam_col; sort=true) : [out]
    for sub in groups
        idx = sort(parentindices(sub)[1])
        t = Float64.(out[idx, t_col])
        m = Float64.(out[idx, mag_col])
        e = Float64.(out[idx, err_col])

        base_s = rolling_time_median(t, m; days=days_short, min_points=min_points, past_only=true)
        base_l = rolling_time_median(t, m; days=days_long, min_points=min_points, past_only=true)

        choose_short = isfinite.(base_s) .& isfinite.(base_l) .& (abs.(base_s .- base_l) .> 0.05)
        baseline = similar(base_s)
        for i in eachindex(baseline)
            if choose_short[i] && isfinite(base_s[i])
                baseline[i] = base_s[i]
            elseif isfinite(base_l[i])
                baseline[i] = base_l[i]
            else
                baseline[i] = base_s[i]
            end
        end

        resid = m .- baseline

        e_med = isfinite.(e) |> any ? median(e[isfinite.(e)]) : 0.0
        sigma_loc = rolling_time_mad(t, resid; days=days_short, min_points=max(8, div(min_points, 2)), past_only=true, add_err=e_med)

        tmax = maximum(t[isfinite.(t)])
        near_end = (tmax .- t) .<= last_window_guard
        if any(isnan, sigma_loc[near_end])
            r_good = isfinite.(resid)
            if any(r_good)
                med_r = median(resid[r_good])
                mad_r = mad(resid[r_good]; center=med_r, normalize=true)
                robust = sqrt(max(mad_r, 0.0)^2 + max(e_med, 0.0)^2)
                sigma_loc[near_end .& .!isfinite.(sigma_loc)] .= max(robust, 1e-6)
            end
        end

        sigma_loc = isfinite.(sigma_loc) .* sigma_loc .+ .!isfinite.(sigma_loc) .* 1e-6
        sigma_resid = resid ./ sigma_loc

        out[idx, :baseline] .= baseline
        out[idx, :resid] .= resid
        out[idx, :sigma_resid] .= sigma_resid
    end
    out
end

"""
per-camera GP baseline with a simple SE kernel approximation.

Requires GaussianProcesses.jl. If unavailable or the fit fails, falls back to per-camera median.
"""
function per_camera_gp_baseline(
    df::DataFrame;
    sigma::Union{Nothing, Float64}=nothing,
    rho::Union{Nothing, Float64}=nothing,
    q::Float64=0.7, # unused in this simplified kernel
    S0::Union{Nothing, Float64}=nothing,
    w0::Union{Nothing, Float64}=nothing,
    jitter::Float64=DEFAULT_JITTER,
    t_col::Symbol=:JD,
    mag_col::Symbol=:mag,
    err_col::Symbol=:error,
    cam_col::Symbol=Symbol("camera#"),
    sigma_floor::Union{Nothing, Float64}=nothing,
    floor_clip::Float64=3.0,
    floor_iters::Int=3,
    min_floor_points::Int=30,
    add_sigma_eff_col::Bool=true,
)
    out = copy(df)
    cols = (:baseline, :resid, :sigma_resid, :baseline_source)
    for col in cols
        if !(col in names(out))
            out[!, col] = col == :baseline_source ? fill("unknown", nrow(out)) : fill(NaN, nrow(out))
        end
    end
    if add_sigma_eff_col && !(:sigma_eff in names(out))
        out[!, :sigma_eff] = fill(NaN, nrow(out))
    end

    function _robust_sigma_floor(resid, yerr_here, var_here)
        finite0 = isfinite.(resid) .& isfinite.(yerr_here) .& isfinite.(var_here)
        if count(finite0) < max(10, min_floor_points)
            return 0.0
        end
        r = copy(resid[finite0])
        keep = trues(length(r))
        for _ in 1:max(floor_iters, 1)
            rr = r[keep]
            if length(rr) < max(10, min_floor_points)
                break
            end
            med = median(rr)
            mad_val = mad(rr; center=med, normalize=true)
            mad_val = max(mad_val, 1e-12)
            keep = abs.(r .- med) .<= floor_clip * mad_val
        end
        rr = r[keep]
        if length(rr) < max(10, min_floor_points)
            rr = r
        end
        s_quiet = mad(rr; center=median(rr), normalize=true)
        s_quiet = max(s_quiet, 1e-12)

        yerr2_med = median((yerr_here[finite0][keep]).^2)
        var_med = median(var_here[finite0][keep])
        floor2 = max(s_quiet^2 - yerr2_med - var_med, 0.0)
        sqrt(floor2)
    end

    groups = cam_col in names(out) ? groupby(out, cam_col; sort=true) : [out]
    for sub in groups
        idx = sort(parentindices(sub)[1])
        t = Float64.(out[idx, t_col])
        y = Float64.(out[idx, mag_col])
        yerr = err_col in names(out) ? Float64.(out[idx, err_col]) : fill(NaN, length(idx))

        finite = isfinite.(t) .& isfinite.(y)
        if count(finite) < 5
            if any(isfinite.(y))
                baseline_val = median(y[isfinite.(y)])
                baseline = fill(baseline_val, length(y))
                resid = y .- baseline
                med_yerr_all = any(isfinite.(yerr)) ? median(yerr[isfinite.(yerr)]) : jitter
                yerr_full = replace(yerr, x -> isfinite(x) ? x : med_yerr_all)
                yerr_full = max.(yerr_full, 0.0)
                sigma_eff = sqrt.(yerr_full .^ 2 .+ jitter^2)
                sigma_resid = resid ./ sigma_eff
                out[idx, :baseline] .= baseline
                out[idx, :resid] .= resid
                out[idx, :sigma_resid] .= sigma_resid
                if add_sigma_eff_col
                    out[idx, :sigma_eff] .= sigma_eff
                end
                out[idx, :baseline_source] .= "median_fallback"
            end
            continue
        end

        finite_idx = findall(finite)
        t_fit = t[finite_idx]
        y_fit = y[finite_idx]
        y_mean = mean(y_fit)
        y_centered = y_fit .- y_mean

        yerr_fit = yerr[finite_idx]
        if !any(isfinite.(yerr_fit))
            yerr_fit = fill(jitter, length(y_fit))
        else
            med_yerr = median(yerr_fit[isfinite.(yerr_fit)])
            med_yerr = isfinite(med_yerr) ? med_yerr : jitter
            yerr_fit = replace(yerr_fit, x -> isfinite(x) ? x : med_yerr)
        end
        yerr_fit = max.(yerr_fit, 0.0)

        baseline = fill(NaN, length(y))
        var = zeros(length(y))
        baseline_flag = "median_fallback"

        try
            @eval using GaussianProcesses
            ell = rho === nothing ? 200.0 : rho
            amp = sigma === nothing ? 0.05 : sigma
            k = GaussianProcesses.SEIso(amp, ell)
            X = reshape(t_fit, :, 1)
            gp = GaussianProcesses.GP(X, y_centered, GaussianProcesses.MeanZero(), k, yerr_fit .^ 2 .+ jitter^2)
            μ, Σ = GaussianProcesses.predict_f(gp, reshape(t, :, 1))
            baseline .= vec(μ) .+ y_mean
            var .= clamp.(diag(Σ), 0.0, Inf)
            baseline_flag = "gp_se"
        catch err
            @warn "GP fit failed; falling back to median baseline" err
            y_med = median(y[finite])
            baseline .= y_med
            var .= 0.0
            baseline_flag = "median_fallback"
        end

        resid = y .- baseline

        med_yerr_all = any(isfinite.(yerr)) ? median(yerr[isfinite.(yerr)]) : jitter
        yerr_full = replace(yerr, x -> isfinite(x) ? x : med_yerr_all)
        yerr_full = max.(yerr_full, 0.0)

        floor_here = sigma_floor === nothing ? _robust_sigma_floor(resid, yerr_full, var) : max(sigma_floor, 0.0)
        sigma_eff = sqrt.(max.(yerr_full .^ 2 .+ floor_here^2 .+ var, 1e-12))
        sigma_resid = resid ./ sigma_eff

        out[idx, :baseline] .= baseline
        out[idx, :resid] .= resid
        out[idx, :sigma_resid] .= sigma_resid
        if add_sigma_eff_col
            out[idx, :sigma_eff] .= sigma_eff
        end
        out[idx, :baseline_source] .= baseline_flag
    end
    out
end

"""
Masked per-camera GP baseline: drops dip windows before fitting GP (simplified kernel).
"""
function per_camera_gp_baseline_masked(
    df::DataFrame;
    dip_sigma_thresh::Float64=-1.0,
    pad_days::Float64=100.0,
    S0::Float64=0.0005,
    w0::Float64=0.0031415926535897933,
    Q::Float64=0.7,
    a1=nothing, rho1=nothing, a2=nothing, rho2=nothing,
    jitter::Float64=DEFAULT_JITTER,
    use_yerr::Bool=true,
    t_col::Symbol=:JD,
    mag_col::Symbol=:mag,
    err_col::Symbol=:error,
    cam_col::Symbol=Symbol("camera#"),
    min_gp_points::Int=10,
)
    out = copy(df)
    for col in (:baseline, :resid, :sigma_resid)
        if !(col in names(out))
            out[!, col] = fill(NaN, nrow(out))
        end
    end

    groups = cam_col in names(out) ? groupby(out, cam_col; sort=true) : [out]
    for sub in groups
        idx = sort(parentindices(sub)[1])
        t = Float64.(out[idx, t_col])
        y = Float64.(out[idx, mag_col])
        yerr = if use_yerr && err_col in names(out)
            Float64.(out[idx, err_col])
        else
            fill(NaN, length(idx))
        end

        finite = isfinite.(t) .& isfinite.(y)

        y_med = median(y[finite])
        r0 = y .- y_med
        r0_f = r0[finite]
        med_r = median(r0_f)
        mad_r = mad(r0_f; center=med_r, normalize=true)

        e_med = if use_yerr && any(isfinite.(yerr))
            median(yerr[finite .& isfinite.(yerr)])
        else
            jitter
        end

        s0 = sqrt(max(mad_r, 0.0)^2 + max(e_med, 0.0)^2)
        s0 = max(s0, 1e-6)

        sig0 = r0 ./ s0
        dip_flag = finite .& isfinite.(sig0) .& (sig0 .< dip_sigma_thresh)

        keep = copy(finite)
        if any(dip_flag)
            t_dip = t[dip_flag]
            bad = falses(length(keep))
            for td in t_dip
                bad .|= abs.(t .- td) .<= pad_days
            end
            keep .&= .!bad
        end

        if count(keep) < min_gp_points
            baseline = fill(y_med, length(y))
            resid = y .- baseline
            out[idx, :baseline] .= baseline
            out[idx, :resid] .= resid
            out[idx, :sigma_resid] .= resid ./ s0
            continue
        end

        t_fit = t[keep]
        y_fit = y[keep]

        yerr_fit = if use_yerr && any(isfinite.(yerr[keep]))
            yerr_tmp = yerr[keep]
            med = median(yerr_tmp[isfinite.(yerr_tmp)])
            replace(yerr_tmp, x -> isfinite(x) ? x : med)
        else
            fill(jitter, length(y_fit))
        end
        yerr_fit = replace(yerr_fit, x -> isfinite(x) ? x : jitter)

        y_mean = mean(y_fit)
        y_fit0 = y_fit .- y_mean

        baseline = similar(y)
        sigma_resid = similar(y)
        try
            @eval using GaussianProcesses
            ell = rho1 === nothing ? 200.0 : rho1
            amp = a1 === nothing ? 0.05 : a1
            k = GaussianProcesses.SEIso(amp, ell)
            gp = GaussianProcesses.GP(reshape(t_fit, :, 1), y_fit0, GaussianProcesses.MeanZero(), k, yerr_fit .^ 2 .+ jitter^2)
            μ, Σ = GaussianProcesses.predict_f(gp, reshape(t, :, 1))
            baseline .= vec(μ) .+ y_mean
            var = clamp.(diag(Σ), 0.0, Inf)
            med_err = median(yerr_fit)
            scale = sqrt.(max.(var .+ med_err^2, 1e-12))
            sigma_resid .= (y .- baseline) ./ scale
        catch err
            baseline .= y_med
            resid = y .- baseline
            sigma_resid .= resid ./ s0
        end

        out[idx, :baseline] .= baseline
        out[idx, :resid] .= y .- baseline
        out[idx, :sigma_resid] .= sigma_resid
    end
    out
end

end # module
