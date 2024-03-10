module pd_density

using Images, Optim    # Images is a pretty heavy dependency. Can you use a subset? I typically start with ImageCore and then add specific needed functonality.
using FFTW, ZernikePolynomials, DSP

struct InitialParam
    n::Float64
    NA::Float64
    lambda::Float64
    Z_orders::Int64
end

export zernike_img_fit

include("zernike_fit.jl")
include("suppzern.jl")
include("pd_initialize.jl")

end
