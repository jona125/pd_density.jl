# Pupil initialization  pd_initial()
# Zernike initialization zern_initial()
#

function pd_initial(NA, lambda, imsz)
    pupil = NA / lambda

    H = zeros(imsz[1:2])
    xrng = yrng = [1:ceil(imsz[1] / 2); -floor(imsz[1] / 2):-1] ./ imsz[1] * 4

    theta = [atan(y, x) for y in yrng, x in xrng]
    rho = [hypot(x, y) for y in yrng, x in xrng]

    H = rho .^ 2 .<= pupil^2

    return H, rho, theta
end

function zern_initial(H, rho, initial_param, imsz)
    (; n, NA, lambda, Z_orders, pixel_spacing) = initial_param
    z_ratio = pixel_spacing[3] / minimum(pixel_spacing[1:2])
    z = imsz[3] * z_ratio
    zFrame = (1:z_ratio:z) .- Int(floor(z / 2))
    zFrame = abs.(zFrame)

    coef = n / lambda
    kz = sqrt.(Complex.(coef^2 .- rho .^ 2))
    Hz = zeros(Complex{Float64}, imsz)
    for ix in eachindex(zFrame)
        Hz[:, :, ix] = H .* cispi.(2 * zFrame[ix] .* kz)
    end
    return Hz
end
