begin
    using NPZ
    using Plots

    function photon_sensitivity(IM, I0, pupil)
        # Compute reference intensities ( = photon noise)
        I0_norm = vec(I0) ./ sum(I0)
        # Compute photon_sensitivity
        photon_loss = 1 # sum(I0) ./ sum(abs2, pupil) # take in account lost photons in the WFS
        A = IM ./ sqrt.(I0_norm)
        replace!(A, NaN => 0.0)
        s = sqrt.(photon_loss) .* sqrt.(sum(A.^ 2,dims=1))
        return s
    end

    pupil = npzread("data/sensitivity_test/aperture_efield.npy")
    I0 = npzread("data/sensitivity_test/lantern_flat.npy")
    IM = npzread("data/sensitivity_test/lantern_im.npy")

    photon_sensitivity(IM, I0, pupil)
    plot(photon_sensitivity(IM, I0, pupil)[1,:] * (1.55e-6) / (4π))
end