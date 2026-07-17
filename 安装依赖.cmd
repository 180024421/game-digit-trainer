@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  if errorlevel 1 (
    echo 失败: 请先安装 Python 3.11+ 并加入 PATH
    pause
    exit /b 1
  )
)
call .venv\Scripts\python -m pip install -U pip
call .venv\Scripts\pip install -e ".[dev]"
if errorlevel 1 (
  echo 依赖安装失败
  pause
  exit /b 1
)
echo 依赖安装完成。可双击 一键启动.cmd
pause
