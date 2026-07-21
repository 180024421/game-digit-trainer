@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 请先运行 安装依赖.cmd
  echo 当前目录: %CD%
  exit /b 1
)
".venv\Scripts\python.exe" -c "from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer" 2>nul
if errorlevel 1 (
  echo 正在安装 RapidOCR...
  ".venv\Scripts\pip.exe" install onnxruntime rapidocr-onnxruntime --no-deps
)
if "%~2"=="" (
  echo 用法: 比对标题.cmd 图A.png 图B.png
  echo 推荐: 先运行 比对标题服务.cmd 常驻，再反复比对更快
  echo 可选: --local --debug --json --keyword 武神登场 --roi-a x,y,w,h
  exit /b 2
)
".venv\Scripts\python.exe" tools\match_same_title.py %*
exit /b %ERRORLEVEL%
