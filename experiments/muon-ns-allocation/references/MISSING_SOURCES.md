# 交接材料登记（更新 2026-09-29 16:05）

用户已补齐全部四份材料，现均在 references/（只读）：
1. MUON_PROBLEM_FIRST_PROJECT_BRIEF_v1_CN.md（原立项稿）
2. ROUND1_AUDIT_REPORT_CN.md
3. ROUND2_AUDIT_AND_RESEARCH_ASSESSMENT_CN.md
4. ROUND3_AUDIT_AND_CAUSAL_ANALYSIS_CN.md

早期登记的三份缺失已作废。本轮执行依据 P0_P1_EXECUTION_ADDENDUM_v1_1_CN.md（对原
立项稿的修订：候选数 15->45/输入（三联幅度探针）、同预算联合探针、评价隔离、AMO
配方命名），两份文件冲突处以补充协议为准并已在协议文档列明。

从 ROUND3 审计落实的本轮防护（对 T2b 缺陷的对应修复）：
- 参数参考在 load 后建立（p1_engine.theta_plus 于 opt.step() 后快照；anchor 参考在
  load_anchor 内）；
- 实际使用的梯度做前后 hash（非无关张量）；
- no-op 候选即生产 opt.step() 本身，另以独立第二次 fwd/bwd 采集对照；
- AdamW step 张量跨候选别名检查（load_state_dict 后 data_ptr 断言）。
