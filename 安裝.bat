@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在建立環境並安裝套件，第一次約需幾分鐘...
if not exist .venv python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo 安裝完成！之後雙擊「啟動.bat」即可使用。
pause
