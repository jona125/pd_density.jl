# Pupil initialization  pd_initial()
# Zernike initialization zern_initial()
#

function pd_initial(NA, lambda, imsz)
    pupil = NA / lambda

    H = zeros(imsz[1:2])
    ## START BLOCK
    X = mapreduce(
        permutedims,
        vcat,
        ([[1:imsz[1]/2; -imsz[1]/2:-1] for _ = 1:imsz[2]]) ./ imsz[1] * 4,
    )

    theta = atan.(X'X)
    rho = hypot.(X, X')
    ## END BLOCK
    # This block is not inferrable (it triggers red bars in ProfileView; left-click on the `construct_Zernmat` red bar
    # and then type `descend_clicked()` on the command line, see the ProfileView and Cthulhu documentation).
    # It also looks like an inefficient way to compute `rho` and `theta`, as it involves formation of a temporary `X` and
    # further allocations from its usage. Would something like
    #   xrng = yrng = blah   # you fill in `blah`
    #   theta = [atan(y, x) for y in yrng, x in xrng]
    #   rho = [hypot(x, y) for y in yrng, x in xrng]
    # be correct, more efficient, and inferrable? Comprehensions, as this syntax is called, are almost always better than
    # Matlab-esque ways of building such things.
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
    zFrame = (1:z) .- Int(floor(z / 2))

    coef = n / lambda
    kz = sqrt.(Complex.(coef^2 .- rho .^ 2))
    Hz = zeros(Complex{Float64}, imsz)
    for ix = 1:z
        Hz[:, :, ix] = H .* exp.(im * 2 * pi * zFrame[ix] .* kz)  # see comment in suppzern.jl. Is this change correct?
    end
    return Hz
end
