*   [首页](README.md)
*   背景与协作
    *   [项目元提示词](collaboration/context-and-roles.md)
    *   [AI 对话记录](analysis-dialogue-log.md)
    *   [一个关于侧边栏和"魔法"短语的小故事](collaboration/sidebar-resize-story.md)

*   基础性思考 (Foundational Thinking)
    *   [概述](foundational-thinking/README.md)
    *   [用户信息表征 (三层结构方案)](foundational-thinking/foundational-3layer-user-representation.md)
    *   [核心机制：用户信息注入提示词](foundational-thinking/injecting-user-info-into-prompts.md)
    *   [个性化的价值场景思考](foundational-thinking/personalization-scenario-analysis.md)
    *   [AI 角色定位：工具 vs. 良师益友](foundational-thinking/ai-role-tool-vs-mentor.md)

*  系统设计与实现
    *   [架构设计概览](architecture.md)    <!-- # 重点解释 How (整体) -->
    *   [核心逻辑解读](core-logic-explained.md)   <!-- # 重点解释 What & Why -->
    *   [个性化价值场景分析](foundational-thinking/personalization-scenario-analysis.md) <!-- New analysis doc -->
    *   HMM 层级详解 (HMM Layer Details) 
        *   [L0: 输入处理与洞察生成](layer-analysis/L0-deep-dive.md)
        *   [L1: 特征提取与结构化记忆](layer-analysis/L1-deep-dive.md)
        *   [L2: 对齐微调与记忆整合](layer-analysis/L2-deep-dive.md)
        *   [HMM 工作流程示例](layer-analysis/hmm-example-walkthrough.md)
    *   [Prompt 工程分析](analysis/prompt-engineering-deep-dive.md)  <!-- # 实现个性化的关键技术 -->

*  工程实践与质量 (TODO)
    <!-- *   [代码质量](code-quality.md)
    *   [测试策略](testing.md)
    *   [部署运维](deployment.md)
    *   [安全考量](security.md) -->

*   FAQ
    *   [GraphRAG/知识图谱的必要性](FAQ/necessity-of-graphrag.md)
    *   [模型对齐方法 (SFT/DPO/LoRA)](FAQ/alignment-methods.md)
    *   [L0 层的复杂性与成本效益](FAQ/L0-complexity-concerns.md) 