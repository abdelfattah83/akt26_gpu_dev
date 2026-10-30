# Assignments GPU — AKT26
This repo is a fork of abdelfattah83/akt26_gpu_dev. We wrote the GPU code in two files:

3-ger/ger.cu
4-gemm/gemm.cu
## Compilation Windows

Avec un CUDA Toolkit installé et `nvcc` sur `PATH`, les scripts compilent puis
exécutent chaque assignment :

```powershell
powershell -ExecutionPolicy Bypass -File .\3-ger\run.ps1
powershell -ExecutionPolicy Bypass -File .\3-ger\run.ps1 -M 2048 -N 1024 -Repeats 100
powershell -ExecutionPolicy Bypass -File .\3-ger\run.ps1 -M 128 -N 96 -Repeats 10 
.
.
...
powershell -ExecutionPolicy Bypass -File .\4-gemm\run.ps1
powershell -ExecutionPolicy Bypass -File .\4-gemm\run.ps1 -N 32 -BatchSize 256
powershell -ExecutionPolicy Bypass -File .\4-gemm\run.ps1 -N 17 -BatchSize 500      
```

Les scripts ciblent `sm_86` par défaut (Notre GPU:RTX 3050). Les exécutables `.exe` sont générés dans leurs dossiers respectifs.

