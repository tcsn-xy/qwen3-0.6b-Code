$env:PYTHONUNBUFFERED = 1
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
Set-Location C:\all\project\study\qwen-0.6B\t2\t1-过拟合失误
& "C:\Program Files\Python312\python.exe" -u inference.py --mode sft
