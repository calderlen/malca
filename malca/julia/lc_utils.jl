module LcUtils

using DataFrames
using CSV
using Glob
using Printf
using Dates: DateTime
try
    using PyPlot
catch
end

const colors = [
    "#6b8bcd",
    "#b3b540",
    "#8f62ca",
    "#5eb550",
    "#c75d9c",
    "#4bb092",
    "#c5562f",
    "#6c7f39",
    "#ce5761",
    "#c68c45",
    "#b5b246",
    "#d77fcc",
    "#7362cf",
    "#ce443f",
    "#3fc1bf",
    "#cda735",
    "#a1b055",
]

"""
    read_lc_dat2(asassn_id, path)

Load an ASAS-SN light curve from `<path>/<asassn_id>.dat2` if present; otherwise
fall back to CSV files `<asassn_id>-light-curves.csv` or `<asassn_id>.csv`.
Returns a pair `(df_g, df_v)` split by v/g band flag.
"""
function read_lc_dat2(asassn_id, path)
    dat2_path = joinpath(path, "$(asassn_id).dat2")
    if isfile(dat2_path)
        columns = [
            "JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "cam_field",
        ]

        df = DataFrame(
            CSV.File(
                dat2_path;
                header=false,
                delim=' ',
                ignorerepeated=true,
                select=1:length(columns),
                rename=columns,
            ),
        )

        if "cam_field" in names(df)
            tmp = split.(df.cam_field, "/")
            camera_name = [length(x) > 0 ? x[1] : "" for x in tmp]
            field = [length(x) > 1 ? x[2] : "" for x in tmp]
            df[!, :camera_name] = camera_name
            df[!, :field] = field
            select!(df, Not(:cam_field))
        end

        transform!(df, [
            :JD => ByRow(Float64) => :JD,
            :mag => ByRow(Float64) => :mag,
            :error => ByRow(Float64) => :error,
            :good_bad => ByRow(Int) => :good_bad,
            Symbol("camera#") => ByRow(Int) => Symbol("camera#"),
            :v_g_band => ByRow(Int) => :v_g_band,
            :saturated => ByRow(Int) => :saturated,
            :camera_name => ByRow(String) => :camera_name,
            :field => ByRow(String) => :field,
        ])

        df_g = df[df.v_g_band .== 0, :]
        df_v = df[df.v_g_band .== 1, :]
        return df_g, df_v
    end

    csv_candidates = [
        joinpath(path, "$(asassn_id)-light-curves.csv"),
        joinpath(path, "$(asassn_id).csv"),
    ]
    idx_csv = findfirst(isfile, csv_candidates)
    if idx_csv !== nothing
        csv_path = csv_candidates[idx_csv]
        df = DataFrame(CSV.File(csv_path; comment='#', ignorerepeated=true))

        rename_map = Dict(
            "Mag" => "mag",
            "Mag Error" => "error",
            "JD" => "JD",
            "Filter" => "filter",
            "Camera" => "camera",
        )
        for (k, v) in rename_map
            if k in names(df)
                rename!(df, k => v)
            end
        end

        if "camera" in names(df)
            df[!, :camera] = strip.(string.(df.camera))
            df[!, Symbol("camera#")] = df.camera
            df[!, :cam_field] = df[!, Symbol("camera#")]
        else
            df[!, :camera] = ""
            df[!, Symbol("camera#")] = ""
            df[!, :cam_field] = ""
        end

        df[!, :saturated] = haskey(df, :saturated) ? df.saturated : fill(0, nrow(df))

        if "Quality" in names(df)
            df[!, :quality_flag] = uppercase.(strip.(string.(df.Quality)))
            df[!, :good_bad] = Int.(df.quality_flag .== "G")
        else
            df[!, :quality_flag] = fill("G", nrow(df))
            df[!, :good_bad] = fill(1, nrow(df))
        end

        _band_flag(val) = beginswith(uppercase(string(val)), "V") ? 1 : 0
        df[!, :v_g_band] = if "filter" in names(df)
            _band_flag.(df.filter)
        else
            fill(0, nrow(df))
        end

        for col in (:JD, :mag, :error)
            if col in names(df)
                df[!, col] = map(x -> begin
                    v = tryparse(Float64, string(x))
                    v === nothing ? NaN : v
                end, df[!, col])
            end
        end

        df_g = df[df.v_g_band .== 0, :]
        df_v = df[df.v_g_band .== 1, :]
        return df_g, df_v
    end

    @info "[error] $(asassn_id): file not found in $(path)"
    return DataFrame(), DataFrame()
end

"""
    read_lc_raw(asassn_id, path)

Read `<asassn_id>.raw` containing per-camera summary statistics.
"""
function read_lc_raw(asassn_id, path)
    raw_path = joinpath(path, "$(asassn_id).raw")
    if !isfile(raw_path)
        return DataFrame()
    end
    columns = [
        Symbol("camera#"),
        :median,
        :sig1_low,
        :sig1_high,
        :p90_low,
        :p90_high,
    ]
    df = DataFrame(
        CSV.File(
            raw_path;
            delim=' ',
            ignorerepeated=true,
            header=false,
            rename=columns,
            types=Dict(
                Symbol("camera#") => Int,
                :median => Float64,
                :sig1_low => Float64,
                :sig1_high => Float64,
                :p90_low => Float64,
                :p90_high => Float64,
            ),
        ),
    )
    return df
end

"""
    match_index_to_lc(; index_path, lc_path, mag_bins, id_column)

Iterate over `index*_masked.csv` files and emit a vector of NamedTuples with
presence info for corresponding `.dat` files.
"""
function match_index_to_lc(;
    index_path::AbstractString="/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path::AbstractString="/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins::Vector{<:AbstractString}=["12_12.5", "12.5_13", "13_13.5", "13.5_14", "14_14.5", "14.5_15"],
    id_column::AbstractString="asas_sn_id",
)
    re_idx = r"index(\d+)_masked\.csv"i
    results = NamedTuple[]

    for mag_bin in mag_bins
        idx_paths = sort(glob("index*_masked.csv", joinpath(index_path, mag_bin)))
        for idx_csv in idx_paths
            m = match(re_idx, basename(idx_csv))
            isnothing(m) && continue
            idx_num = parse(Int, m.captures[1])
            lc_dir = joinpath(lc_path, mag_bin, "lc$(idx_num)_cal")

            ids_df = DataFrame(CSV.File(idx_csv; types=Dict(id_column => String)))
            ids = unique(dropmissing(ids_df[!, id_column]))

            for asn in ids
                dat_path = joinpath(lc_dir, "$(asn).dat")
                found = isfile(dat_path)
                push!(results, (; mag_bin, index_num=idx_num, index_csv=idx_csv, lc_dir, asas_sn_id=asn, dat_path=found ? dat_path : nothing, found))
            end
        end
    end
    results
end

# Internal helpers to format sexagesimal values
deg_to_hms(ra_deg::Real) = begin
    total_seconds = ra_deg / 15 * 3600
    h = Int(floor(total_seconds / 3600))
    m = Int(floor((total_seconds - h * 3600) / 60))
    s = round(Int, total_seconds - h * 3600 - m * 60)
    h %= 24
    (h, m, s)
end

deg_to_dms(dec_deg::Real) = begin
    sign = dec_deg < 0 ? -1 : 1
    dabs = abs(dec_deg)
    d = Int(floor(dabs))
    m = Int(floor((dabs - d) * 60))
    s = round(Int, (dabs - d - m / 60) * 3600)
    (sign * d, sign * m, sign * s)
end

"""
    custom_id(ra_deg, dec_deg)

Construct a J2000-style identifier from RA/Dec in degrees (e.g., `Jhhmmss+ddmmss`).
"""
function custom_id(ra_deg, dec_deg)
    h, m, s = deg_to_hms(ra_deg)
    dd, dm, ds = deg_to_dms(dec_deg)
    sign = dd < 0 ? "-" : "+"
    dd, dm, ds = abs(dd), abs(dm), abs(ds)
    @sprintf("J%02d%02d%02d%s%02d%02d%02d", h, m, s, sign, dd, dm, ds)
end

"""
    plotparams(ax; labelsize=15)

Mirror of the matplotlib styling helper; operates on a PyPlot axis if available.
"""
function plotparams(ax; labelsize=15)
    @assert @isdefined(PyPlot) "PyPlot not available"
    ax.minorticks_on()
    ax[:yaxis][:set_ticks_position]("both")
    ax[:xaxis][:set_ticks_position]("both")
    ax.tick_params(direction="in", which="both", labelsize=labelsize)
    ax.tick_params("both", length=8, width=1.8, which="major")
    ax.tick_params("both", length=4, width=1, which="minor")
    for axis in ["top", "bottom", "left", "right"]
        ax[:spines][axis][:set_linewidth](1.5)
    end
    ax
end

"""
Placeholder for parity with the Python stub.
"""
function divide_cameras()
    nothing
end

end # module
