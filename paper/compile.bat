@echo off
REM LaTeX Paper Compilation Script
REM For Cross Residual GNN Paper

echo ===================================
echo Compiling LaTeX Paper...
echo ===================================

REM Clean previous compilation files
echo.
echo [1/4] Cleaning previous files...
del /Q *.aux *.log *.out *.bbl *.blg *.synctex.gz 2>nul

REM First pass: pdflatex
echo.
echo [2/4] First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: First pdflatex pass failed!
    pause
    exit /b 1
)

REM Run bibtex
echo.
echo [3/4] Running bibtex...
bibtex main
if errorlevel 1 (
    echo ERROR: bibtex failed!
    pause
    exit /b 1
)

REM Second pass: pdflatex (for citations)
echo.
echo [4/4] Second pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: Second pdflatex pass failed!
    pause
    exit /b 1
)

REM Third pass: pdflatex (for cross-references)
echo.
echo Running final pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 (
    echo ERROR: Final pdflatex pass failed!
    pause
    exit /b 1
)

echo.
echo ===================================
echo Compilation completed successfully!
echo ===================================
echo.
echo Output file: main.pdf
echo.

REM Open the PDF if compilation was successful
if exist main.pdf (
    echo Opening PDF...
    start main.pdf
) else (
    echo WARNING: main.pdf not found!
)

pause
