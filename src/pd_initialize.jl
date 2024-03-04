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

function zern_initial(img, H, rho, initial_param)
    n, _, lambda, _, _ = initial_param
    z = size(img)[3]
    zFrame = (1-Int(z / 2)):Int(z / 2)

    coef = n / lambda
    kz = sqrt.(Complex.(coef^2 .- rho .^ 2))
    Hz = zeros(Complex{Float64}, size(img))
    for ix = 1:z
        Hz[:, :, ix] = H .* exp(im * 2 * pi * zFrame[ix] .* kz)
    end
    return Hz
end
