@echo off
chcp 65001 >nul
rem ============================================================
rem  run_all.bat —— 一键跑完整流程（下载数据 -> 划分 -> 训练 -> 评估）
rem  双击本文件即可；也可以在命令行里单独执行每一步
rem ============================================================
set PY=py -3.12
cd /d %~dp0

echo [1/4] 检查环境与 GPU ...
%PY% -c "import paddle; paddle.utils.run_check()"

echo.
echo [2/4] 下载数据集（已下载会自动跳过）...
%PY% download_dataset.py

echo.
echo [3/4] 划分数据集（8:1:1）...
%PY% prepare_dataset.py

echo.
echo [4/4] 训练模型 ...
%PY% train.py --epochs 10

echo.
echo 训练完成，开始评估 ...
%PY% evaluate.py

echo.
echo 全部完成！接下来可以：
echo   单张图片识别:  %PY% predict.py --image demo_images
echo   Web 界面:      %PY% app.py
echo   实时演示:      %PY% camera_demo.py
pause
