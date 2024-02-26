function zernike_value(H, n_coefs)
    Zval = zeros((size(H)..., n_coefs))
    for ix = 1:n_coefs
        Zval[:, :, ix] = Zern_gen(ix + 2, size(H)) .* H
    end
    return Zval
end

function Zern_gen(p, sz)
    n = Int(ceil((-3 + sqrt(9 + 8 * p)) / 2))
    m = Int(2 * p - n .* (n + 2))

    Z = Zernike(m, n; coord = :cartesian)
    out = [
        Z.(((1:sz[1]) .* 2 .- sz[1]) ./ sz[1], i) for
        i in ((1:sz[2]) .* 2 .- sz[2]) ./ sz[2]
    ]
    return mapreduce(permutedims, vcat, out)
end

function Zcoefs2phi(Zcoefs, Zval)
    @show Zcoefs
    phi = zeros(size(Zval)[1:end-1])
    for i = 1:length(Zcoefs)
        phi = phi + Zcoefs[i] * Zval[:, :, i]
    end
    return phi
end
