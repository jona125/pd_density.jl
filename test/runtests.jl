using pd_density
using Test
using Optim, FiniteDifferences, FFTW, Statistics

function Z_test(Zcoeffs, img, initial_param::pd_density.InitialParam, noise = 0.0)
    img_ = pd_density.construct_Zernimg(Zcoeffs, img, initial_param)
    img_ .+= noise * mean(img_) .* rand(size(img)...)
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Zcoeffs atol = 1e-4
    return result
end

@testset "pd_density.jl" begin

    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 11 # Z(2.-2) -> Z(4,4)

    # test empty image
    img = zeros(32, 32, 32)
    img[16, 16, 16] = 1.0
    initial_param = pd_density.InitialParam(n, NA, lambda, Z_orders)
    Z = zeros(Z_orders)

    # test construct_Zernimg()
    img_ = pd_density.construct_Zernimg(Z, img, initial_param)
    Zk = copy(Z)
    Zk[3] = 3 * lambda
    imgstack = zeros(size(img)..., 2)
    imgstack[:, :, :, 1] = img_
    imgstack[:, :, :, 2] = pd_density.construct_Zernimg(Zk, img_, initial_param, true)

    Hz, Zval, _ = pd_density.construct_Zernmat(initial_param, size(img_))
    Hk, Dk, Sk, _, ukeep, _, _ = pd_density.loss_prep(Z, imgstack, Hz, Zval, Zk)
    F = zeros(Complex{Float64}, (size(img)..., 2))
    for i = 1:2
        for id in findall(ukeep)
            idx = CartesianIndex(id, i)
            F[idx] = (Dk[idx] .* conj(Sk[idx])) / abs2.(Sk[idx])
        end
    end
    #@test img ≈ abs.(ifft(F)) atol = 1
    @test img_ ≈ abs.(ifft(F[:, :, :, 1] .* Sk[:, :, :, 1])) atol = 0.01
    @test imgstack[:, :, :, 2] ≈ abs.(ifft(F[:, :, :, 2] .* Sk[:, :, :, 2])) atol = 0.01

    # test fake img with K phase diversity image
    Zcol = []
    push!(Zcol, copy(Z))
    push!(Zcol, copy(Zk))
    result = zernike_img_fit(imgstack, initial_param; Zcol, g_abstol = 1e-14)


    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4


    # test gradient function
    f(X) = pd_density.psfloss(X, img_, Hz, Zval, img)
    g!(g, X) = pd_density.psfgrad!(g, X, img_, Hz, Zval, img)
    g = zeros(Z_orders)
    #@test g!(g, Zk) ≈ grad(central_fdm(2, 1), f, Zk)[1] atol = eps()

    # test fake img with psf effect
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-14)


    # test fake img with Zernike coefficient
    for i = 1:Z_orders
        Zcoeffs = copy(Z)
        Zcoeffs[i] = 1.0
        Z_test(Zcoeffs, img, initial_param)
        noise = 0.02
        Z_test(Zcoeffs, img, initial_param, noise)
    end

end
