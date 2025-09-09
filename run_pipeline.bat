@echo off
rem Repro pipeline: tosa->linalg, bufferize, linalg->parallel, GPU mapping
set "MLIR_OPT=out\build\x64-debug\llvm\llvm\bin\mlir-opt.exe"

echo Running tosa->linalg...
"%MLIR_OPT%" --pass-pipeline=builtin.module(func.func(tosa-to-linalg)) input_tosa.mlir -o step1.mlir 1>step1_stdout.txt 2>&1
if exist step1.mlir (type step1.mlir) else (type step1_stdout.txt)

echo Running one-shot bufferize...
"%MLIR_OPT%" --one-shot-bufferize step1.mlir -o step2.mlir 1>step2_stdout.txt 2>&1
if exist step2.mlir (type step2.mlir) else (type step2_stdout.txt)

echo Converting linalg to parallel loops...
"%MLIR_OPT%" --pass-pipeline=builtin.module(func.func(convert-linalg-to-parallel-loops)) step2.mlir -o step3_parallel.mlir 1>step3p_stdout.txt 2>&1
if exist step3_parallel.mlir (type step3_parallel.mlir) else (type step3p_stdout.txt)

echo Running GPU mapping pipeline...
"%MLIR_OPT%" --pass-pipeline=builtin.module(func.func(gpu-map-parallel-loops,convert-parallel-loops-to-gpu),gpu-kernel-outlining) step3_parallel.mlir -o step4.mlir 1>step4_stdout.txt 2>&1
if exist step4.mlir (type step4.mlir) else (type step4_stdout.txt)

echo Done.
exit /B 0