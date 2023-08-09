using Plots
using CSV, NPZ, DataFrames, FITSIO
using StatsBase: countmap, mean
using IterTools: partition

function partition_by(f, arr)
    """
    Partitions arr according to the return value of `f` called on each element of arr. Returns a dictionary whose keys are the f values and whose values are the corresponding arr elements sharing that f value.
    """
    fvals = f.(arr)
    d = Dict()
    for v in Set(fvals)
        d[v] = getindex.(Ref(arr), findall(x -> x == v, fvals))
    end
    return d
end

f = FITS("data/shane_control_matrix/reconMatrix_16x/Hw.fits")
interaction_matrix = read(f[1])
close(f)

lantern_files = filter(endswith(".tiff"), readdir("data/pl_230602/onsky1"))
rstamp = r"onsky1-\d{8}(\d{6})-(\d+).tiff"

function isless_tstamp(f1, f2)
    m1, m2 = match(rstamp, f1), match(rstamp, f2)
    if m1[1] < m2[1]
        return true
    else
        return parse(Int, m1[2]) < parse(Int, m2[2])
    end
end

sort!(lantern_files, lt=isless_tstamp)
frames_per_s = countmap(getindex.(match.(rstamp, lantern_files), 1))
files_by_tstamp = partition_by(
    x -> getindex(match(rstamp, x), 1),
    lantern_files
)
for (k, v) in files_by_tstamp
    sort!(files_by_tstamp[k], lt=isless_tstamp)
end

function stamp_to_seconds(stamp)
    h, m, s = parse(Int, stamp[1:2]), parse(Int, stamp[3:4]), parse(Float64, stamp[5:end])
    return 3600 * h + 60 * m + s
end

times_by_second = Dict()
for (k, v) in files_by_tstamp
    stamp = stamp_to_seconds(k)
    n = length(v)
    times_by_second[k] = stamp .+ collect(0:(1/n):((n-1)/(n)))
end

all_times = sort!(vcat(values(times_by_second)...))

rutc = r"2023-06-03T(\d\d):(\d\d):(\d\d.\d+)"

begin
    initial_tstamps = []
    telemetry = Dict()
    for k in 27:86
        telemetry_file = FITS("data/pl_230602/Telemetry--UCOLick_2023-06-02/Data_00$k.fits")
        telemetry[k] = interaction_matrix * read(telemetry_file[1])[1:288,2:end]
        st = match(rutc, read_header(telemetry_file[1])["UTC"])

        push!(
            initial_tstamps, 
            stamp_to_seconds(string(st[1] * st[2] * st[3]))
        )
        close(telemetry_file)
    end
    initial_tstamps .+= 61200 # UTC correction
end

# telemetry_in_zerns = interaction_matrix * telemetry
# times = 0:1e-3:(1e-3*(size(telemetry_in_zerns, 2)-1))
# plot(times, telemetry_in_zerns[1:3,:]', label=["tip" "tilt" "focus"], xlabel="Time (s)", ylabel="Zernike mode (? units)")

function woofer(t_start, t_end)
    """
    Returns the state of the woofer averaged between t_start and t_end, in Zernike space.

    Doesn't look across files, because there's gaps between telemetry files. Instead, it cuts off its average at the end of a file.
    """
    file_index = findlast(x -> x <= t_start, initial_tstamps)
    if isnothing(file_index)
        return -1
    end
    offset = t_start - initial_tstamps[file_index]
    interval = t_end - t_start
    start_index = clamp(Int(round(offset * 1000)), 1, 4096)
    end_index = clamp(Int(round((offset + interval) * 1000)), 1, 4096)
    if start_index == end_index
        return -1
    end
    return mean(telemetry[file_index+26][:,start_index:end_index], dims=2)
end

good_indices = Int64[]
woofers = []
for (i, (ti, tf)) in enumerate(partition(all_times, 2, 1))
    w = woofer(ti, tf)
    if w != -1
        push!(good_indices, i)
        push!(woofers, w)
    end
end

lantern_intensities = npzread("data/pl_230602/onsky1/onsky1_intensities.npy")[good_indices,:]

inputzs = Matrix(transpose(hcat(woofers...)))
outputls = lantern_intensities
npzwrite("data/pl_230602/inputzs_onsky1.npy", inputzs)
npzwrite("data/pl_230602/outputls_onsky1.npy", outputls)