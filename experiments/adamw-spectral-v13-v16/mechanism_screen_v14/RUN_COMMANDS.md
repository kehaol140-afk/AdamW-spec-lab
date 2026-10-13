# V14 已执行命令与续跑边界

项目根目录：`E:\大模型预训练\adamw_spectral_pilot`。解释器：`D:\pytorch\anaconda\envs\py39\python.exe`。PowerShell 环境变量：`$env:USE_TF='0'; $env:PYTHONDONTWRITEBYTECODE='1'`。

在全新 V14 子目录中按序执行：

```powershell
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m unittest mechanism_screen_v14.test_screen -v
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.screen prepare
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.screen dev
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.screen select
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.screen confirm
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.screen finalize
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m mechanism_screen_v14.posthoc_audit
```

`RUN_VERIFIED.ps1` 是全新运行的固定顺序包装。当前完成后的目录包含冻结选择，故不可从头重跑 `dev` 或打开确认集；只读复核可运行 `screen status` 与 `posthoc_audit`。若在未完成单元中断，`dev`/`confirm` 可使用已写入的原始行继续，同时保留累计预算；本次 Windows 原子文件替换重试的旧源码、旧 manifest、预算与原始行快照保存在 `attempts/atomic_retry_2/`。

实际运行有一次启动 RAM 阻断（`attempts/pre_gpu_revision_1/logs/`）和一次 WinError 5 原子预算替换失败（`attempts/atomic_retry_2/dev_failure.log`）。二者发生在相应下一次前向之前。无 optimizer.step 调用。
