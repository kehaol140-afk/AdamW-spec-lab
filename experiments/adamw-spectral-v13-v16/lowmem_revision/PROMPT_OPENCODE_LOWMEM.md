# 发给 OpenCode：审核低内存修订后恢复原实验

工作目录 `E:\大模型预训练\adamw_spectral_pilot`。
先读本目录下lowmem_revision/LOWMEM_AMENDMENT.md与reports/LOWMEM_HANDOFF.md，再实际审核代码、活动config/local_v1_lowmem.json、CPU测试、GPU smoke/replay和资源日志。核对只有资源门禁/日志/版本交接变化，科学设置不变。不要只复述Codex报告。

该修订允许运行RAM硬线由2降至1GiB，并设置初始化前1.5GiB准入、运行中1.5GiB警告。不要仍按旧2GiB判停，也不要越过新1GiB硬线。不得把新准入线重复应用于正在正常运行的每步。检查初始化/训练/checkpoint/probe都受保护，阶段资源采样与实际步数可追溯。

先确认项目独占、无另一个训练进程和有效锁冲突。不要杀其他用户进程，不升级环境、不下载数据、不改模型或科学合同。

若最新真实smoke PASS与活动代码/config/binding/data指纹一致，直接复用，不重复smoke占用12步。若不匹配，先核算20步smoke上限，不自动重试直到超预算。资源监控模拟测试不能替代真实smoke。

随后使用实际绑定解释器，串行执行：
`run.py pilot --config config/local_v1_lowmem.json --stage discover`
只有local_gate.json指纹匹配且next_action=RUN_CONFIRM时，再执行同配置confirm。

发现负科学结果不是软件错误，不改rank、层、LR范围或统计门槛救场。运行中RAM<1GiB或RSS/显存超限就停止；记录准确完成步数和最后有效checkpoint，不能声称未保存的步已经持久化。禁止漏算重放/重试预算。不自动扩展短分支或服务器任务。

阶段结束输出 reports/LOWMEM_OPENCODE_AUDIT.md、reports/LOCAL_FINDINGS.md、原始CSV/JSON、每步资源日志、真正的峰值/最小RAM及最新版轻量包。若需用户释放后台内存，说明缺少多少与哪一阶段，不继续降低阈值。Windows换页观测不足时标记NOT_MEASURED，不能写“无换页”。
