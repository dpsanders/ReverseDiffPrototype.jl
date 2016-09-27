#################
# TapeNode/Tape #
#################

immutable TapeNode{F,I,O,M}
    func::F
    inputs::I
    outputs::O
    cache::M # holds data used in reverse pass (gradients, dual arrays, value arrays, etc.)
end

typealias Tape Vector{TapeNode}

function record!(tp::Nullable{Tape}, func, inputs, outputs, cache = nothing)
    !(isnull(tp)) && push!(get(tp), TapeNode(func, inputs, outputs, cache))
    return nothing
end

###################
# Pretty Printing #
###################

compactrepr(x, _...) = repr(x)
compactrepr(x::AbstractArray, _...) = match(r"\[.*?\]", repr(x)).match

function compactrepr(t::Tuple, pad = "")
    io = IOBuffer()
    print(io, "(")
    print(io, compactrepr(t[1]))
    for i in drop(t, 1)
        println(io, ",")
        print(io, " ", pad, compactrepr(i))
    end
    print(io, ")")
    return takebuf_string(io)
end

function Base.show(io::IO, node::TapeNode, pad = "")
    println(io, pad, "TapeNode($(node.func)):")
    # length of the prefix strings below (e.g. "  inputs:  ")
    # plus whatever extra padding was passed in
    valpad = repeat(" ", 11 + length(pad))
    println(io, pad, "  inputs:  ", compactrepr(node.inputs, valpad))
    println(io, pad, "  outputs: ", compactrepr(node.outputs, valpad))
    print(io,   pad, "  cache:   ", compactrepr(node.cache, valpad))
end

Base.display(tp::Tape) = show(STDOUT, tp)

function Base.show(io::IO, tp::Tape)
    println("$(length(tp))-element Vector{TapeNode}:")
    for node in tp
        println("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        show(io, node)
        println()
    end
end
