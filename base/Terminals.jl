# This file is a part of Julia. License is MIT: http://julialang.org/license

module Terminals

export
    TextTerminal,
    UnixTerminal,
    TerminalBuffer,
    TTYTerminal,
    cmove,
    cmove_col,
    cmove_down,
    cmove_left,
    cmove_line_down,
    cmove_line_up,
    cmove_right,
    cmove_up,
    disable_bracketed_paste,
    enable_bracketed_paste,
    end_keypad_transmit_mode,
    getX,
    getY,
    hascolor,
    pos,
    raw!,
    writepos

import Base:
    flush,
    read,
    readuntil,
    iosize,
    start_reading,
    stop_reading,
    write,
    writemime,
    reseteof,
    eof,
    check_open # stream.jl

## TextTerminal ##

abstract TextTerminal <: Base.IO

# INTERFACE
iosize(::TextTerminal) = error("Unimplemented")
writepos(t::TextTerminal, x, y, s::Array{UInt8,1}) = error("Unimplemented")
cmove(t::TextTerminal, x, y) = error("Unimplemented")
getX(t::TextTerminal) = error("Unimplemented")
getY(t::TextTerminal) = error("Unimplemented")
pos(t::TextTerminal) = (getX(t), getY(t))
reseteof(t::TextTerminal) = nothing

# Relative moves (Absolute position fallbacks)
cmove_up(t::TextTerminal, n) = cmove(getX(t), max(1, getY(t)-n))
cmove_up(t) = cmove_up(t, 1)

cmove_down(t::TextTerminal, n) = cmove(getX(t), max(height(t), getY(t)+n))
cmove_down(t) = cmove_down(t, 1)

cmove_left(t::TextTerminal, n) = cmove(max(1, getX(t)-n), getY(t))
cmove_left(t) = cmove_left(t, 1)

cmove_right(t::TextTerminal, n) = cmove(max(width(t), getX(t)+n), getY(t))
cmove_right(t) = cmove_right(t, 1)

cmove_line_up(t::TextTerminal, n) = cmove(1, max(1, getY(t)-n))
cmove_line_up(t) = cmove_line_up(t, 1)

cmove_line_down(t::TextTerminal, n) = cmove(1, max(height(t), getY(t)+n))
cmove_line_down(t) = cmove_line_down(t, 1)

cmove_col(t::TextTerminal, c) = cmove(c, getY(t))

# Defaults
hascolor(::TextTerminal) = false

# Utility Functions
function writepos{T}(t::TextTerminal, x, y, b::Array{T})
    if isbits(T)
        writepos(t, x, y, reinterpret(UInt8, b))
    else
        cmove(t, x, y)
        invoke(write, Tuple{IO, Array}, s, a)
    end
end
function writepos(t::TextTerminal, x, y, args...)
    cmove(t, x, y)
    write(t, args...)
end
width(t::TextTerminal) = iosize(t)[2]
height(t::TextTerminal) = iosize(t)[1]

# For terminals with buffers
flush(t::TextTerminal) = nothing

clear(t::TextTerminal) = error("Unimplemented")
clear_line(t::TextTerminal, row) = error("Unimplemented")
clear_line(t::TextTerminal) = error("Unimplemented")

raw!(t::TextTerminal, raw::Bool) = error("Unimplemented")

beep(t::TextTerminal) = nothing
enable_bracketed_paste(t::TextTerminal) = nothing
disable_bracketed_paste(t::TextTerminal) = nothing

## UnixTerminal ##

abstract UnixTerminal <: TextTerminal

type TerminalBuffer <: UnixTerminal
    out_stream::Base.IO
end

type TTYTerminal <: UnixTerminal
    term_type::ASCIIString
    in_stream::Base.TTY
    out_stream::Base.TTY
    err_stream::Base.TTY
end

reseteof(t::TTYTerminal) = reseteof(t.in_stream)

const CSI = "\x1b["

cmove_up(t::UnixTerminal, n) = write(t.out_stream, "$(CSI)$(n)A")
cmove_down(t::UnixTerminal, n) = write(t.out_stream, "$(CSI)$(n)B")
cmove_right(t::UnixTerminal, n) = write(t.out_stream, "$(CSI)$(n)C")
cmove_left(t::UnixTerminal, n) = write(t.out_stream, "$(CSI)$(n)D")
cmove_line_up(t::UnixTerminal, n) = (cmove_up(t, n); cmove_col(t, 0))
cmove_line_down(t::UnixTerminal, n) = (cmove_down(t, n); cmove_col(t, 0))
cmove_col(t::UnixTerminal, n) = write(t.out_stream, "$(CSI)$(n)G")

@windows ? begin
    function raw!(t::TTYTerminal,raw::Bool)
        check_open(t.in_stream)
        if Base.ispty(t.in_stream)
            run(if raw
                    `stty raw -echo onlcr -ocrnl opost`
                else
                    `stty sane`
                end,t.in_stream,t.out_stream,t.err_stream)
            true
        else
            ccall(:jl_tty_set_mode,
                 Int32, (Ptr{Void},Int32),
                 t.in_stream.handle, raw) != -1
        end
    end
end : begin
    function raw!(t::TTYTerminal, raw::Bool)
        check_open(t.in_stream)
        ccall(:jl_tty_set_mode, Int32, (Ptr{Void},Int32), t.in_stream.handle, raw) != -1
    end
end
enable_bracketed_paste(t::UnixTerminal) = write(t.out_stream, "$(CSI)?2004h")
disable_bracketed_paste(t::UnixTerminal) = write(t.out_stream, "$(CSI)?2004l")
end_keypad_transmit_mode(t::UnixTerminal) = # tput rmkx
    write(t.out_stream, "$(CSI)?1l\x1b>")

function Base.iosize(t::UnixTerminal)
    return iosize(t.out_stream)
end

clear(t::UnixTerminal) = write(t.out_stream, "\x1b[H\x1b[2J")
clear_line(t::UnixTerminal) = write(t.out_stream, "\x1b[0G\x1b[0K")
#beep(t::UnixTerminal) = write(t.err_stream,"\x7")

write{T,N}(t::UnixTerminal, a::Array{T,N}) = write(t.out_stream, a)
write(t::UnixTerminal, p::Ptr{UInt8}) = write(t.out_stream, p)
write(t::UnixTerminal, p::Ptr{UInt8}, x::Integer) = write(t.out_stream, p, x)
write(t::UnixTerminal, x::UInt8) = write(t.out_stream, x)
read{T,N}(t::UnixTerminal, x::Array{T,N}) = read(t.in_stream, x)
readuntil(t::UnixTerminal, s::AbstractString) = readuntil(t.in_stream, s)
readuntil(t::UnixTerminal, c::Char) = readuntil(t.in_stream, c)
readuntil(t::UnixTerminal, s) = readuntil(t.in_stream, s)
read(t::UnixTerminal, ::Type{UInt8}) = read(t.in_stream, UInt8)
start_reading(t::UnixTerminal) = start_reading(t.in_stream)
stop_reading(t::UnixTerminal) = stop_reading(t.in_stream)
eof(t::UnixTerminal) = eof(t.in_stream)

@unix_only hascolor(t::TTYTerminal) = (startswith(t.term_type, "xterm") || success(`tput setaf 0`))
@windows_only hascolor(t::TTYTerminal) = true

end # module
