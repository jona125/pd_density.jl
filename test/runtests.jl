using pd_density
using Test
using Optim, ZernikePolynomials, FFTW
include("../src/pd_initialize.jl")
include("../src/suppzern.jl")

function construct_Zern(Zcoefs, initial_param, img)
    n, NA, lambda, imsz, Z_orders = initial_param
    H, rho, theta = pd_initial(NA, lambda, imsz)
    Zval = zernike_value(H, Z_orders, rho, theta)
    Hz = zern_initial(img, H, rho, initial_param)

    _, Sk = ZernFF(Zcoefs, Hz, Zval, imsz)

    imgfft = fft(img)

    return abs.(ifft(imgfft .* Sk))
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
    stack = create_sphere(stack, (128, 128, 128), 96, 128)
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
    initial_param = n, lambda, NA, size(img), Z_orders
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4

    img = generate_fake_img()
    initial_param = n, lambda, NA, size(img), Z_orders
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4

    img = generate_fake_img()
    Z = zeros(1, Z_orders)
    Z[2] = 0.3
    img_ = construct_Zern(Z, initial_param, img)
    img_ = img_./maximum(img_) * 0.5 + img
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4


    Z = rand(1, Z_orders)
    img_ = construct_Zern(Z, initial_param, img)
    img_ = img_./maximum(img_) * 0.5 + img
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4

end
