# 发给 Codex：最小资源修订，不重做实验

在 `E:\大模型预训练\adamw_spectral_pilot` 工作。阅读本文件同目录 LOWMEM_AMENDMENT.md、config/local_v1_lowmem.json 与 evidence/，再实际读取现有报告和源码。保持原MiniMind、原数据、绑定解释器只读。任务是把已经获准的小实验从过严的系统RAM门禁中恢复，不是重新设计研究。

1. 核查本地是否仍与上传版本一致，是否有本项目进程、run.lock或新的正式结果。不要假定已关闭，不能自动杀进程。先归档旧代码、配置、日志、报告与manifest，不复制1.24GB原数据，不删除缓存来源。保留config/local_v1.json；把提供的低内存配置置于config/local_v1_lowmem.json。
2. 采用协议ADAMW_A_SPECTRAL_LOCAL_V1_1_LOWMEM_20261002。运行RAM硬线1.0 GiB；新训练进程重初始化前准入1.5 GiB；初始化后警告线1.5 GiB。精确定义见AMENDMENT，不要把1.5的准入线重复用于每步。RSS 4GiB、显存保护、smoke保护不变。新字段要实际读取并测试，不能只写JSON。
3. 先添加失败的资源门禁单测，再最小修复。统一pilot_core.py、pilot_workflow.py、prepare_data.py中的阈值检查和硬编码“2 GiB”文本。记录初始化前/后、首个opt.step、每步、checkpoint及probe边界的系统available RAM、RSS、CUDA allocated/reserved/peak、实际步数和版本哈希。不要提前初始化CUDA来采“初始化前”的数据，不要用mock值当实测。
4. 不动模型、batch、precision、LR、moments、裁剪、eta*wd、数据拆分、谱/标量网格、主cell、阈值或数值容差。先不做候选生成器、snapshot或offload重构。原资源门禁行为也要保留兼容，不得globally替换全部2为1。
5. 同步源码指纹两份SOURCE_FILES、活动protocol/config记录、数据manifest和bundle_light（原来仅打包旧local_v1.json）。如需prepare，以原绑定源窗口确定性重建，并核对token/eval/doc_roles内容SHA保持一致；不手填哈希以绕过检查。科学设置必须与原配置完全相同。
6. 先CPU原测试+新增测试，再真实GPU smoke/replay。总实际GPU smoke优化器步数仍≤20；正常8步+4重放=12。更新前RAM阻断可在条件变化后最多重试2次；更新后失败不可不计账地重跑。错误原因/完成步数必须从真实执行记录得出，不能被run.py异常处理覆盖丢失。
7. 不跑discover/confirm，不安装/升级/下载，不关闭别人的程序，不改页面文件，不把1GiB降到0.5GiB，不伪造smoke PASS。若仍阻断，报告实际初始化后余量与阶段，停止。

交付 reports/LOWMEM_HANDOFF.md、配置/科学字段差异、代码diff、全部单测日志、真实GPU资源曲线与峰值、最新smoke/replay产物和原始退出码、新版轻量包、后续绑定解释器命令。写明是否真的进行了12个GPU更新、是否通过replay；没有实际完成不得声称修复成功。
