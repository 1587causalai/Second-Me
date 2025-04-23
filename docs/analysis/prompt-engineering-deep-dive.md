# Prompt 工程深度解析

## 项目背景与 Prompt 的核心作用

`Second-Me` 项目旨在构建一个用户的个性化 AI "数字镜像"，它能够理解用户的知识体系、模拟用户的口吻并遵循用户的偏好。实现这一目标的核心依赖于其分层记忆模型 (HMM: L0, L1, L2) 以及对大型语言模型 (LLM) 的精巧运用。

**Prompt 工程** 在这个架构中扮演着至关重要的角色。它如同与 LLM 对话的"剧本"和"导演"，在 HMM 的每一层以及相关的处理流程中，精确地引导 LLM 完成特定任务：

*   **L0 层:** 通过 Prompt 指导 LLM 理解多样化的单次用户输入，并生成初步的结构化洞察 (`Note`)。
*   **L1 层:** 利用精心设计的 Prompt，引导 LLM 对聚合后的记忆进行深度分析，提取抽象特征（如 `Topics`, `Shades`）并构建结构化的用户画像 (`Bio`)。
*   **L2 层:** 通过特定的系统提示 (System Prompts) 和指令格式，为 SFT (Supervised Fine-Tuning) 和 DPO (Direct Preference Optimization) 准备训练数据，将 L1 的结构化记忆"注入"基础模型，实现个性化对齐。
*   **辅助流程 (如 GraphRAG):** 使用 Prompt 指导 LLM 或相关工具从用户数据中提取特定的结构化信息（如实体、关系）。

本文档的目标就是深入分析这些遍布于项目代码中的 Prompt，理解它们的设计意图、结构特点以及如何共同协作，最终驱动 `Second-Me` 实现其个性化目标。我们将逐层、逐模块地进行解析。

---

## L0 层 Prompts (初步洞察生成)

L0 层针对不同输入类型（图像、音频、文档）采用了多阶段、专门化的 Prompt 策略，旨在从单次输入中提取结构化信息并生成初步洞察。

*   **关键文件:** `lpm_kernel/L0/prompt.py`, `lpm_kernel/L0/l0_generator.py`
*   **核心特点:**
    *   **类型特定:** 为图像 (`insight_image_*`)、音频 (`insight_audio_*`)、文档 (`insight_doc_*`) 设计了不同的 Prompt。
    *   **多阶段处理:** 复杂输入（图、音）采用解析、概览、分解等多步 Prompt 调用。
    *   **角色扮演:** 定义清晰的角色（如图像分类助手、老朋友、音频洞察专家）。
    *   **结构化输出 (JSON):** 绝大多数 Prompt 强制要求输出可解析的 JSON 格式，包含特定字段（如 Title, Overview, Breakdown, Insight 等）。
    *   **上下文整合:** 部分 Prompt (如 `insight_image_overview`) 包含占位符，用于整合用户 Bio 信息。
    *   **详细指令:** 包含明确的 Workflow 和 Guidelines，精确控制 LLM 行为。

*   **示例 Prompt (图像处理):**
    *   `insight_image_parser`: 分类图像为情感或知识型。
    *   `insight_image_overview`: 以朋友口吻结合 Hint 和 Bio 生成标题和开场白。
    *   `insight_image_breakdown`: 以朋友口吻生成多个深入的、带背景知识的洞察点。

*   **总结:** L0 Prompt 设计高度专业化和结构化，通过任务分解和精确指令，从异构输入中可靠地提取初步洞察，为 L1 提供标准化的 `Note` 输入。

## L1 层 Prompts (特征提取与结构化记忆)

L1 层使用 Prompt 引导 LLM 对 L0 产生的 `Note` 进行聚合分析，提取抽象特征并构建核心用户画像 (`Bio`)。

*   **关键文件:** `lpm_kernel/L1/prompt.py`, `lpm_kernel/L1/topics_generator.py`, `lpm_kernel/L1/shade_generator.py`, `lpm_kernel/L1/l1_generator.py`

### 1. 主题生成 (TopicsGenerator)

*   **Prompts:** `TOPICS_TEMPLATE_SYS`, `TOPICS_TEMPLATE_USR`, `SYS_COMB`, `USR_COMB`
*   **任务:**
    *   `TOPICS_TEMPLATE_SYS/USR`: 为聚类后的记忆块生成简洁的主题名称 (`topic`) 和关键词标签 (`tags`)。
    *   `SYS_COMB/USR_COMB`: (可能用于后处理) 合并相似的主题。
*   **特点:**
    *   相对简洁，聚焦于归纳总结。
    *   强制 JSON 输出 (`topic`, `tags`)。
    *   输入为具体的记忆/笔记内容。

### 2. Shade 生成与处理 (ShadeGenerator & ShadeMerger)

*   **Prompts:** `SHADE_INITIAL_PROMPT`, `SHADE_IMPROVE_PROMPT`, `PERSON_PERSPECTIVE_SHIFT_V2_PROMPT`, `SHADE_MERGE_PROMPT`, `SHADE_MERGE_DEFAULT_SYSTEM_PROMPT`
*   **任务:**
    *   `SHADE_INITIAL_PROMPT`: 从零开始分析记忆，生成代表用户某方面特征的 `ShadeInfo` (含 Name, Aspect, Icon, Desc, Content, Timelines)。
    *   `SHADE_IMPROVE_PROMPT`: 基于新记忆更新已有的 `ShadeInfo`。
    *   `PERSON_PERSPECTIVE_SHIFT_V2_PROMPT`: 将 `ShadeInfo` 中的第三人称描述转换为第二人称。
    *   `SHADE_MERGE_PROMPT`: 将多个相似的 `ShadeInfo` 合并成一个新的、更泛化的 `ShadeInfo`。
    *   `SHADE_MERGE_DEFAULT_SYSTEM_PROMPT`: (供 `ShadeMerger` 调用) 判断哪些 Shade ID 可以被合并。
*   **特点:**
    *   **专家角色扮演:** 要求 LLM 扮演数据分析/心理学专家。
    *   **高度结构化:** 严格依赖结构化输入，并强制要求输出特定格式的 JSON。
    *   **任务分解:** 将复杂的 Shade 构建分解为初始化、改进、视角转换、合并判断、合并执行等步骤。
    *   **上下文利用:** 明确要求利用提供的记忆或 Timeline 信息。

### 3. Bio 生成 (L1Generator 调用)

*   **Prompt:** `GLOBAL_BIO_SYSTEM_PROMPT`
*   **任务:** 基于 L1 生成的多个 `ShadeInfo` (结构化特征)，构建一个**全面的、多维度的用户画像文本报告**。
*   **要求:**
    *   进行**深度推断和推测**（个性特质、兴趣分布、可能职业/身份）。
    *   输出**简洁的自然语言文本**（<200 字），而非 JSON。
*   **特点:**
    *   L1 生成的**最高层综合**步骤。
    *   要求高层抽象和概括能力。
    *   输出为面向 L2 的、高度概括的用户画像描述。

*   **总结:** L1 Prompt 设计精巧，通过多步骤、结构化的方式引导 LLM 从离散的 `Note` 中提取、抽象、转换并整合信息，最终构建出包含深层特征 (`Shades`) 和全局概括 (`Bio`) 的结构化用户记忆。

## GraphRAG 相关 Prompts

GraphRAG 流程使用 Prompt 从文本中提取结构化信息，用于增强 L2 训练数据。

*   **关键文件:** `lpm_kernel/L2/data_pipeline/graphrag_indexing/prompts/extract_graph.txt`, `summarize_descriptions.txt`, `lpm_kernel/L2/data.py` (`graphrag_indexing` 方法)
*   **Prompts:**
    *   `extract_graph.txt`:
        *   **任务:** 根据预定义类型列表 (`{entity_types}`)，从输入文本 (`{input_text}`) 中提取实体（名称、类型、描述）和实体间的关系（源、目标、描述、强度）。
        *   **特点:** 格式要求极其严格（特定元组结构、分隔符），强调英文输出（特定词除外）。
    *   `summarize_descriptions.txt`:
        *   **任务:** 将关于同一实体的多个描述 (`{description_list}`) 合并成一个单一、全面、连贯的摘要。
        *   **特点:** 要求解决信息冲突，使用第三人称和英文。
*   **总结:** GraphRAG Prompts 专注于**结构化信息提取**，其严格的格式控制是为了生成干净、可靠的实体和关系数据，供 L2 数据生成流程使用。

## L2 层 Prompts (SFT/DPO 个性化对齐)

L2 层使用系统提示来指导 SFT 训练，并使用评估 Prompt 来驱动 DPO 的偏好数据生成。

*   **关键文件:** `lpm_kernel/L2/training_prompt.py`, `lpm_kernel/L2/dpo/prompt.py`, `lpm_kernel/L2/utils.py`, `lpm_kernel/L2/dpo/dpo_data.py`

### 1. SFT 系统提示 (`utils.create_chat_data` 调用)

*   **核心身份 Prompt:** `MEMORY_PROMPT` / `MEMORY_COT_PROMPT`: 定义模型为用户的 "Second Me" 或 "Me.bot"，任务是基于用户背景和记录回答问题。COT 版本要求结构化思考 (`<think>`, `<answer>`)。
*   **特定任务 Prompts:**
    *   `CONTEXT_PROMPT` / `CONTEXT_COT_PROMPT`: 指导模型学习如何基于用户理解来**丰富和澄清**用户的初始需求。
    *   `JUDGE_PROMPT` / `JUDGE_COT_PROMPT`: 指导模型学习如何**代表用户评估专家回复**并提供反馈。
*   **特点:**
    *   **角色与任务导向:** 根据训练数据类型赋予不同角色和任务。
    *   **CoT 广泛应用:** 通过结构化思考引导提升输出质量。
    *   **个性化嵌入:** `{user_name}` 占位符。
*   **总结:** SFT 系统提示旨在教会模型**扮演好 "Second Me" 的角色**，在不同场景下根据用户背景做出恰当的回应。

### 2. DPO 相关 Prompt (AI Judge - `dpo_data.py` 调用)

*   **评估 Prompts:** `MEMORY_EVAL_SYS`, `CONTEXT_ENHANCE_EVAL_SYS`, `JUDGE_EVAL_SYS`
*   **任务:** 指导一个强大的 AI Judge LLM (如 GPT-4o) 评估两个候选模型响应的优劣。
*   **特点:**
    *   **目标驱动:** 清晰定义了每个评估任务的目标。
    *   **多维度标准:** 评估标准非常细致全面（准确性、个性化、相关性、帮助性、共情性、角色一致性等）。
    *   **依赖参考信息:** 明确要求结合用户画像、笔记等背景信息进行判断。
    *   **严肃性强调:** 反复强调评估的重要性。
    *   **中文分析输出:** 要求详细分析使用中文。
*   **通用输入模板:** `USR` 模板规范了提交给 AI Judge 的输入格式。
*   **总结:** DPO 评估 Prompt 设计复杂且要求严格，旨在自动化生成高质量的偏好数据，驱动 L2 模型学习更符合用户期望的行为模式。

## 整体结论

`Second-Me` 项目的 Prompt 工程体现了**分层、模块化、结构化**的设计思想。通过为 HMM 各层及关键流程定制专门的 Prompt，并结合角色扮演、详细指令、结构化输入输出（尤其是 JSON）、CoT 等技术，系统地引导 LLM 完成从理解原始输入、构建抽象记忆到最终实现个性化对齐的复杂任务链条。Prompt 的质量和精巧程度直接关系到整个 HMM 架构能否有效运作并达成其"数字镜像"的目标。

---

## 附录：端到端 Prompt 协作示例

为了更具体地展示不同 Prompt 如何协同工作，我们追踪一个简单的用户想法如何在 HMM 系统中流转：

**用户输入 (文本):** "我应该研究一下如何用 Python 进行异步编程，特别是 asyncio 库，这对提高 I/O 密集型任务的效率很有帮助。"

**1. L0 层：初步理解与结构化 (由 `L0Generator` 处理)**
*   **涉及 Prompt (推测):** L0 内部用于处理文本输入的 Prompt。
*   **Prompt 作用:** 指导 LLM 分析文本，提取关键信息（意图、技术、原因）并生成结构化 `Note`。
*   **输出:** 标准化的 `Note` 对象，包含标题、内容、初步洞察、关键词等。

**2. L1 层：特征提取与记忆构建 (由 `TopicsGenerator`, `ShadeGenerator`, `L1Generator` 等处理)**
*   **a. 主题生成 (`TopicsGenerator`)**
    *   **涉及 Prompt:** `TOPICS_TEMPLATE_SYS`, `TOPICS_TEMPLATE_USR`
    *   **Prompt 作用:** 指导 LLM 为包含此 Note 的记忆聚类生成主题名称和标签。
    *   **输出:** 如 `{"topic": "Python 编程与性能优化", "tags": [...]}`。
*   **b. Shade 生成/更新 (`ShadeGenerator`)**
    *   **涉及 Prompt:** `SHADE_INITIAL_PROMPT` 或 `SHADE_IMPROVE_PROMPT`
    *   **Prompt 作用:** 指导 LLM 分析 Note，生成或更新代表"技术学习者"特征的 `ShadeInfo` JSON。
*   **c. Shade 视角转换 (`ShadeGenerator`)**
    *   **涉及 Prompt:** `PERSON_PERSPECTIVE_SHIFT_V2_PROMPT`
    *   **Prompt 作用:** 指导 LLM 将 `ShadeInfo` 中的描述转为第二人称。
*   **d. Bio 生成 (`L1Generator`)**
    *   **涉及 Prompt:** `GLOBAL_BIO_SYSTEM_PROMPT`
    *   **Prompt 作用:** 指导 LLM 基于所有 `ShadeInfo`，进行高层综合，生成简洁的用户画像文本报告。

**3. GraphRAG 流程：离线实体提取 (由 `L2DataProcessor` 触发)**
*   **涉及 Prompt:** `extract_graph.txt`, `summarize_descriptions.txt`
*   **Prompt 作用:**
    *   `extract_graph.txt`: 指导提取结构化的实体元组 (如 "Python", "asyncio") 和关系。
    *   `summarize_descriptions.txt`: (如果需要) 指导合并关于同一实体的多条描述。
*   **输出:** 包含实体及其关联 Note ID 的结构化数据 (JSON/Parquet)。

**4. L2 数据流水线：生成 SFT/DPO 样本 (由 `L2DataProcessor` 协调)**
*   **a. 多样性数据 (`DiversityDataGenerator`)**
    *   **涉及 Prompt (间接):** 生成的 Q&A 样本是为了适配 SFT 的 `MEMORY_PROMPT`。
    *   **Prompt 作用 (隐式):** 利用 GraphRAG 提取的实体 "asyncio" 和关联 Note 内容，生成针对性的问答对。
*   **b. 上下文需求数据 (`ContextGenerator`)**
    *   **涉及 Prompt (间接):** 生成的样本是为了适配 SFT 的 `CONTEXT_PROMPT`。
    *   **Prompt 作用 (隐式):** 利用实体 "asyncio" 和关联 Note 内容，生成模拟的用户需求澄清或增强。
*   **c. DPO 偏好数据 (`dpo_data.py`)**
    *   **涉及 Prompt:** `MEMORY_EVAL_SYS` (或对应的 EVAL_SYS), `USR`
    *   **Prompt 作用:** 指导 AI Judge LLM 对两个关于 asyncio 的候选回答进行评估，结合参考信息（原始 Note, Bio）判断优劣，生成偏好 (`chosen`/`rejected`)。

**5. L2 训练：模型对齐 (由 `train.py` 执行)**
*   **a. SFT 训练**
    *   **涉及 Prompt:** `MEMORY_PROMPT` (或 CoT 版本) 用于**格式化**步骤 4a 生成的 Q&A 数据，添加系统提示和角色标记。
    *   **Prompt 作用:** 将样本转换为 SFTTrainer 可用的对话格式字符串。
    *   **训练:** `SFTTrainer` 使用格式化数据进行微调。
*   **b. DPO 训练**
    *   **涉及 Prompt (间接):** DPO 训练直接使用步骤 4c 生成的偏好对。AI Judge 使用的评估 Prompt 间接影响了训练方向。
    *   **训练:** `DPOTrainer` (推测) 使用偏好对调整模型，使其更倾向于生成"更好"的回答。

**最终结果:** 经过整个由 Prompt 精心编排的流程，最终的个性化模型能够生成反映用户具体兴趣（如对 asyncio 的关注）的回答。 