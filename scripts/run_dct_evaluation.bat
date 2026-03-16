@echo off
REM Full pipeline: watermark with DCT -> run attacks -> evaluate visible + DCT
set ROOT=%~dp0..
cd /d %ROOT%

set INPUT=data\input\sample.mp4
set WATERMARKED=data\output\watermarked_with_dct.mp4
set POSITIONS=data\output\watermarked_with_dct.positions.json
set ATTACKS_DIR=data\attacks_dct
set CSV_OUT=data\output\robustness_dct_results.csv

if not exist %INPUT% (
    echo Input video not found: %INPUT%
    exit /b 1
)

echo [1/3] Watermarking with visible + DCT...
python scripts/watermark_with_dct.py

echo.
echo [2/3] Running attacks...
python src/run_attacks.py --input %WATERMARKED% --out_dir %ATTACKS_DIR%

echo.
echo [3/3] Evaluating visible + DCT under attacks...
python scripts/evaluate_attacks.py --attacks_dir %ATTACKS_DIR% --pos_json %POSITIONS% --csv_out %CSV_OUT% --global_fallback

echo.
echo Done. Results: %CSV_OUT%
