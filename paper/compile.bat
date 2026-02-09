@echo off
REM Windows batch file for compiling LaTeX document

echo Compiling LaTeX document...

REM Compile with XeLaTeX (recommended)
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex

echo.
echo Compilation complete!
echo Output: main.pdf
echo.
echo To view the PDF, open main.pdf
