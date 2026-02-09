@echo off
REM Windows batch file for compiling LaTeX document

echo Compiling LaTeX document...

REM Compile with pdflatex (compatible with bibtex)
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo.
echo Compilation complete!
echo Output: main.pdf
echo.
echo To view the PDF, open main.pdf
