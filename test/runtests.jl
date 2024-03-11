using pd_density
using Test
using Optim, FiniteDifferences, FFTW

function construct_Zernimg(Zcoefs, img, initial_param::pd_density.InitialParam)
    Hz, Zval = pd_density.construct_Zernmat(img, initial_param)
    _, Sk = pd_density.ZernFT(Zcoefs, Hz, Zval, size(img))


    Dk = fft(img)
    S2tot = abs2.(Sk)
    ukeep = abs.(S2tot) .> eps()
    DdotS = Dk .* conj(Sk)

    F = zeros(Complex{Float64}, size(DdotS))
    for id in findall(ukeep)
        F[id] = DdotS[id] / S2tot[id]
    end

    return abs.(ifft(F))
end

function create_sphere(stack, center, radius, gray_level)
    for coord in CartesianIndices(stack)
        if (
            (coord[1] - center[1])^2 + (coord[2] - center[2])^2 + (coord[3] - center[3])^2
        ) <= radius^2
            stack[coord] += gray_level / 256
        end
    end
    return stack
end

function generate_fake_img()
    stack = zeros(256, 256, 256)
    stack = create_sphere(stack, (128, 128, 128), 96, 256)
    stack = convert(Array{Float64}, stack)
    return stack
end


@testset "pd_density.jl" begin

    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 11 # Z(2.-2) -> Z(4,4)

    img = zeros(128, 128, 128)
    img[64, 64, 64] = 1
    initial_param = pd_density.InitialParam(n, NA, lambda, Z_orders)
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4

    img = generate_fake_img()
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4

    Z = zeros(1, Z_orders)
    Z[2] = 0.3
    img_ = construct_Zernimg(Z, img, initial_param_2)
    img_ = img_ ./ maximum(img_) * 0.5 + img
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4


    Z = rand(1, Z_orders)
    img_ = construct_Zernimg(Z, img, initial_param_2)
    img_ = img_ ./ maximum(img_) * 0.5 + img
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4

end
