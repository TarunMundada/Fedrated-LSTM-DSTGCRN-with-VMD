@echo off
echo Starting CHI-taxi Federated VMD-GAT-Hybrid Experiment
echo ============================================================

REM --- Configuration ---
set SCRIPT_PATH=main.py
set DATASET_NAME=CHI-taxi-FL-VMD-GAT-Hybrid
set SAVE_DIR=results\chi-taxi_federated_vmd_gat

REM --- Pre-flight Checks (Adapt as necessary) ---
if not exist "%SCRIPT_PATH%" (
    echo Error: %SCRIPT_PATH% not found in current directory
    pause
    exit /b 1
)
if not exist "DATA\TransportModes\CHI-taxi\tripdata_full.csv" (
    echo Error: DATA\TransportModes\CHI-taxi\tripdata_full.csv not found
    pause
    exit /b 1
)

REM --- Setup Directories ---
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"

echo.
echo Running Experiment: VMD (K=3) + Hybrid GAT + GCRN
echo.

REM --- Run Python Script (Hybrid GAT+GCRN mode) ---
REM --vmd_k 3: Enables VMD decomposition with K=3 modes.
REM --use_gat 1: Enables GAT spatial modeling.
REM --use_gcrn 1: Enables GCRN spatial modeling (creates a Hybrid model).
REM --gat_heads 2: Uses 2 heads for GAT (matching common config).
REM --gat_fusion_mode concat: Concatenates and projects GAT and GCRN outputs.
REM --clients 3: Enables Federated Learning mode.
python "%SCRIPT_PATH%" ^
    --trip_csv "DATA/TransportModes/CHI-taxi/tripdata_full.csv" ^
    --dataset "%DATASET_NAME%" ^
    --rounds 10 ^
    --clients 3 ^
    --tin 12 ^
    --tout 3 ^
    --epochs_per_round 2 ^
    --batch_size 16 ^
    --save_dir "%SAVE_DIR%" ^
    --vmd_k 3 ^
    --use_gat 1 ^
    --use_gcrn 1 ^
    --gat_heads 2 ^
    --gat_fusion_mode "concat" ^
    --save_vmd 1

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✓ Experiment VMD-GAT-Hybrid completed successfully!
    echo    Results saved to: "%SAVE_DIR%"
) else (
    echo ✗ Experiment VMD-GAT-Hybrid failed with error code: %ERRORLEVEL%
)

echo.
echo Press any key to exit...
pause >null