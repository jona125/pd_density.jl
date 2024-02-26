function pd_initial(NA, lambda, imsz)
    pupil = NA / lambda

    H = zeros(imsz[1:2])
    ori = Int.(ceil.((imsz[1] / 2, imsz[2] / 2)))
    rho = [
        sqrt((i[1] - ori[1])^2 + (i[2] - ori[2])^2) / minimum(imsz) for
        i in CartesianIndices(H)
    ]
    H = rho .^ 2 .<= pupil^2

    return H, rho
end

function zern_initial(img, H, rho, initial_param)
    n, _, lambda, _, _ = initial_param
    zFrame = 1:size(img)[3]

    coef = n / lambda
    kz = sqrt.(Complex.(coef^2 .- rho .^ 2))
    Hz = zeros(Complex{Float64}, size(img))
    for ix = 1:size(img)[3]
        Hz[:, :, ix] = H .* exp(im * 2 * pi * zFrame[ix] .* kz)
    end
    return Hz
end
