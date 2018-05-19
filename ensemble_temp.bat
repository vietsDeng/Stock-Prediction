@echo off
set times=100
for /l %%i in (1,1,%times%) do (
   python D://workspace//Mine//main.py
)

pause
