pdflatex2 () {
    pdflatex $1
    pdflatex $1
}

bibpdflatex2 () {
    local 2="${2:-$1}"  # if $2 is empty, use $1
    pdflatex $1
    biber    $2 
    pdflatex $1
    pdflatex $1
}
