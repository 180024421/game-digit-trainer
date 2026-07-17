@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [1/2] 正在创建虚拟环境并安装依赖（首次较慢，含 PyTorch）...
  python -m venv .venv
  if errorlevel 1 (
    echo 失败: 请先安装 Python 3.11+ 并加入 PATH
    pause
    exit /b 1
  )
  call .venv\Scripts\python -m pip install -U pip
  call .venv\Scripts\pip install -e ".[dev]"
  if errorlevel 1 (
    echo 依赖安装失败
    pause
    exit /b 1
  )
)
echo 正在启动 GUI...
call .venv\Scripts\python -m game_digit_trainer gui
if errorlevel 1 (
  echo.
  echo 启动失败，请查看上方报错
  pause
  exit /b 1
)
