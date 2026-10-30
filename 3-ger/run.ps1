param(
    [int]$M = 4000,
    [int]$N = 2000,
    [int]$Repeats = 100,
    [int]$Arch = 86
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc is unavailable. Install the CUDA Toolkit and add nvcc to PATH."
}

Push-Location $PSScriptRoot
try {
    $archOption = "-arch=sm_$Arch"
    & nvcc -O2 -std=c++17 $archOption ger.cu -o ger.exe -lcublas
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed with exit code $LASTEXITCODE."
    }

    & .\ger.exe $M $N $Repeats
    if ($LASTEXITCODE -ne 0) {
        throw "GER test failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
