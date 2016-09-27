Base.eltype{V}(::GradientResult{V}) = V
Base.eltype{V,G}(::Type{GradientResult{V,G}}) = V
Base.eltype{V}(::JacobianResult{V}) = eltype(V)
Base.eltype{V,J}(::Type{JacobianResult{V,J}}) = eltype(V)
Base.eltype{V}(::HessianResult{V}) = V
Base.eltype{V,G,H}(::Type{HessianResult{V,G,H}}) = V

#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

# utilities #
#-----------#

load_grad_result!(out, result, xt) = adjoint!(out, xt)

function load_grad_result!(out::GradientResult, result, xt)
    out.value = value(result)
    adjoint!(out.gradient, xt)
    return out
end

# gradient #
#----------#

function gradient(f, x, tp::Tape = Tape(), xt = track(x, tp))
    result = f(xt)
    seed!(result)
    backprop!(tp)
    return adjoint(xt)
end

function gradient(f, xs::Tuple, tp::Tape = Tape(),
                  xtrs::Tuple = map(x -> track(x, tp), xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tp)
    return map(adjoint, xtrs)
end

# gradient! #
#-----------#

function gradient!(out, f, x, tp::Tape = Tape(), xt = track(eltype(out), x, tp))
    result = f(xt)
    seed!(result)
    backprop!(tp)
    return load_grad_result!(out, result, xt)
end

function gradient!(outs::Tuple, f, xs::Tuple, tp::Tape = Tape(),
                   xtrs::Tuple = map((out, x) -> track(eltype(out), x, tp), outs, xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tp)
    for i in eachindex(outs)
        load_grad_result!(outs[i], result, xtrs[i])
    end
    return outs
end

######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

# utilities #
#-----------#

function load_jacobian!(out, xt, yt, tp::Tape)
    outmatrix = reshape(out, length(yt), length(xt))
    for i in eachindex(yt)
        n = yt[i]
        seed!(n)
        backprop!(tp)
        for j in eachindex(xt)
            out[i, j] = adjoint(xt[j])
        end
        unseed!(tp)
    end
    return out
end

load_jac_result!(out, xt, yt, tp) = load_jacobian!(out, xt, yt, tp)

function load_jac_result!(out::JacobianResult, xt, yt, tp)
    value!(out.value, yt)
    load_jacobian!(out.jacobian, xt, yt, tp)
    return out
end

# jacobian #
#----------#

function jacobian(f, x, tp::Tape = Tape(), xt = track(x, tp))
    yt = f(xt)
    out = similar(yt, eltype(x), length(yt), length(x))
    return load_jacobian!(out, xt, yt, tp)
end

function jacobian(f, xs::Tuple, tp::Tape = Tape(),
                  xtrs::Tuple = map(x -> track(x, tp), xs))
    yt = f(xtrs...)
    outs = map(x -> similar(yt, eltype(x), length(yt), length(x)), xs)
    for i in eachindex(outs)
        load_jacobian!(outs[i], xtrs[i], yt, tp)
    end
    return outs
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, tp::Tape = Tape(), xt = track(eltype(out), x, tp))
    tp = get(tape(first(xt)))
    yt = f(xt)
    return load_jac_result!(out, xt, yt, tp)
end

function jacobian!(outs::Tuple, f, xs::Tuple, tp::Tape = Tape(),
                   xtrs::Tuple = map((out, x) -> track(eltype(out), x, tp), outs, xs))
   yt = f(xtrs...)
   for i in eachindex(outs)
       load_jac_result!(outs[i], xtrs[i], yt, tp)
   end
   return outs
end

#####################################################
# Hessian of `f(::AbstractArray...)::AbstractArray` #
#####################################################

# hessian #
#---------#

hessian(f, x, args...) = jacobian(y -> gradient(f, y), x, args...)

hessian(f, xs::Tuple, args...) = jacobian((ys...) -> gradient(f, ys), xs, args...)

# hessian! #
#----------#

hessian!(out, f, x, args...) = jacobian!(out, y -> gradient(f, y), x, args...)

hessian!(outs::Tuple, f, xs::Tuple, args...) = jacobian!(outs, (ys...) -> gradient(f, ys), xs, args...)

# function hessian!(out::HessianResult, f, x, args...)
#     outgrad = GradientResult(out.value, out.gradient)
#     jacobian!(out.hessian, y -> gradient!(outgrad, f, y), x, args...)
#     out.value = outgrad.value
#     return out
# end
