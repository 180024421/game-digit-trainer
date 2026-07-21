@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 安装依赖.cmd
  exit /b 1
)
".venv\Scripts\python.exe" -c "from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer" 2>nul
if errorlevel 1 (
  echo 正在安装 RapidOCR...
  ".venv\Scripts\pip.exe" install onnxruntime rapidocr-onnxruntime --no-deps
)
echo 启动标题比对常驻服务（保持此窗口打开）...
".venv\Scripts\python.exe" tools\match_same_title.py --serve
