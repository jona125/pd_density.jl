# Pupil initialization  pd_initial()
# Zernike initialization zern_initial()
#

function pd_initial(NA, lambda, imsz)
    pupil = NA / lambda

    H = zeros(imsz[1:2])
    X = mapreduce(
        permutedims,
        vcat,
        ([[1:imsz[1]/2; -imsz[1]/2:-1] for _ = 1:imsz[2]]) ./ imsz[1] * 4,
    )

    theta = atan.(X'X)
    rho = hypot.(X, X')
    H = rho .^ 2 .<= pupil^2

    # pdk = zeros(size(H)..., length(zk))
    # abr = Zern_gen(4, size(H))

    # for i = 1:length(zk)
    #     pdk = H .* abr .* zk(i)
    # end

    return H, rho, theta
end

function zern_initial(H, rho, initial_param, imsz)
    (; n, NA, lambda, Z_orders) = initial_param
    z = imsz[3]
    zFrame = (1-Int(z / 2)):Int(z / 2)

    coef = n / lambda
    kz = sqrt.(Complex.(coef^2 .- rho .^ 2))
    Hz = zeros(Complex{Float64}, imsz)
    for ix = 1:z
        Hz[:, :, ix] = H .* exp(im * 2 * pi * zFrame[ix] .* kz)
    end
    return Hz
end
