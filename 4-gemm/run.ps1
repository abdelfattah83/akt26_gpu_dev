param(
    [int]$N = 16,
    [int]$BatchSize = 1000,
    [int]$Arch = 86
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc is unavailable. Install the CUDA Toolkit and add nvcc to PATH."
}

Push-Location $PSScriptRoot
try {
    $archOption = "-arch=sm_$Arch"
    & nvcc -O2 -std=c++17 $archOption gemm.cu -o gemm.exe -lcublas
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed with exit code $LASTEXITCODE."
    }

    & .\gemm.exe $N $BatchSize
    if ($LASTEXITCODE -ne 0) {
        throw "GEMM test failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
