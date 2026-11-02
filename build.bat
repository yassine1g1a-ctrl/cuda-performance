@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;%PATH%
nvcc -O3 -arch=sm_75 -o bin\%~n1.exe %1
