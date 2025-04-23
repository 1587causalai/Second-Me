# L2 层深度解析：个性化对齐与模型微调

L2 层是 HMM (Hierarchical Memory Model) 的最终目标实现层，负责将 L1 构建的结构化记忆和用户表征，转化为一个能够深度模拟用户个性、偏好和知识体系的语言模型。其核心技术是利用 **SFT (Supervised Fine-Tuning)** 和 **DPO (Direct Preference Optimization)** 对基础 LLM 进行个性化对齐。

## 1. 核心目标

L2 的核心目标是实现 **深度个性化**。它旨在训练出一个"Second Me"，使其：

*   **理解用户知识:** 掌握用户独特的知识领域和信息结构。
*   **模拟用户口吻:** 使用用户习惯的语言风格和表达方式。
*   **遵循用户偏好:** 在回答和决策中体现用户的价值观和偏好。
*   **保持长期一致性:** 维持统一的人格和记忆。

## 2. 数据流水线回顾与 L2 输入

L2 的训练依赖于一个精心设计的数据准备流水线：

*   **输入源:** 主要来自 L1 层的输出（如 `Note` 列表、`Bio` 对象、`Topics` 信息）以及用户的基本信息。
*   **L2 数据处理器 (`lpm_kernel/L2/data.py:L2DataProcessor`)**:
    *   调用 **GraphRAG** (`graphrag_indexing`) 对 L1 (或更早) 的笔记进行索引，提取实体及其关联的笔记 ID (`entitys_path`, `graph_path`)。
    *   调用多个数据生成器 (`DiversityDataGenerator`, `ContextGenerator`, `PreferenceQAGenerator`, `SelfQA`)，利用 L1 输出和 GraphRAG 提取的实体信息，生成多种类型的训练数据样本（问答、模拟需求、偏好对、自问自答等）。
    *   将所有生成的数据合并成一个或多个 JSON 文件（例如 `merged.json`），作为 SFT 和 DPO 的基础输入。

## 3. SFT 数据准备与格式化

Supervised Fine-Tuning (SFT) 的目标是让模型学会"如何说"，即模仿期望的输出格式和内容。

*   **数据来源:** 主要来自 `L2DataProcessor` 生成的合并数据文件 (`merged.json`)，其中包含了各种问答对、指令跟随等样本。
*   **格式化函数 (`lpm_kernel/L2/utils.py:create_chat_data`)**:
    *   读取合并后的 JSON 数据。
    *   使用 `preprocess` 子函数将每条数据转换为包含角色（system, user, assistant）和内容的对话列表 (`messages`)。
        *   根据数据类型（标准问答、上下文增强、裁判反馈），选择不同的系统提示 (`MEMORY_PROMPT`, `CONTEXT_PROMPT`, `JUDGE_PROMPT` 或其 CoT 版本），并嵌入用户名以个性化。
    *   调用 `tokenizer.apply_chat_template` 将对话列表转换为模型可以直接处理的、带有特殊标记的格式化字符串。
*   **关键技术 (`trl.DataCollatorForCompletionOnlyLM`)**: 在 SFT 训练时，使用此 Data Collator **只计算模型在生成助手回答部分时的损失**，忽略提示部分，提高了训练效率和效果。
*   **输出:** 生成符合 SFTTrainer 要求的 Hugging Face `Dataset` 对象。

## 4. DPO 数据准备

Direct Preference Optimization (DPO) 的目标是让模型学会"选择什么"，即根据偏好数据调整模型，使其更倾向于生成"好"的回答，而非"坏"的回答。

*   **数据来源:** 同样基于 `L2DataProcessor` 生成的数据，但侧重于需要进行偏好判断的样本。
*   **核心逻辑 (`lpm_kernel/L2/dpo/dpo_data.py:preprocess`)**:
    *   构建基础 Prompt (包含系统提示和用户输入)。
    *   **生成候选答案:** 可能使用基础模型或一个候选模型生成一个回答 (`response`)。
    *   **获取参考答案:** 从原始数据中获取一个参考答案 (`reference`)。
    *   **AI Judge 评估 (`compare_eval`)**: 将 Prompt、候选答案、参考答案发送给一个独立的 **AI Judge LLM** 进行评估，判断哪个答案更好。
    *   **生成偏好对:** 根据 AI Judge 的判断，确定哪个是 `chosen`（胜者）回答，哪个是 `rejected`（败者）回答。
*   **输出:** 生成包含 `(prompt, chosen, rejected)` 三元组的偏好数据集，用于 DPO 训练。

## 5. 训练执行与协调

L2 的训练过程由 `lpm_kernel/train/trainprocess_service.py:TrainProcessService` 统一协调：

*   **流程管理:** 使用 `TrainProgressHolder` 跟踪从数据准备到模型转换的每一步状态。
*   **触发方式:** 服务类通过构造**命令行参数**并启动**子进程**的方式，调用 `lpm_kernel/L2/train.py` 脚本来执行实际的训练。
*   **训练脚本 (`lpm_kernel/L2/train.py`)**:
    *   使用 Hugging Face `transformers`, `peft`, `trl` 库。
    *   调用 `utils.create_and_prepare_model` 加载模型和 Tokenizer，支持 LoRA 和量化。
    *   主要执行 **SFT 训练**，使用 `SFTTrainer` 和 `DataCollatorForCompletionOnlyLM`。
    *   **DPO 训练的执行**: 虽然 `train.py` 未直接展示 `DPOTrainer`，但推测 `TrainProcessService` 可以通过传递不同的命令行参数（如指定 DPO 数据集路径或模式参数）给 `train.py` 或调用另一个专用脚本来执行 DPO。
*   **监控:** 通过读取训练脚本输出的日志文件来监控进度。
*   **后续步骤:** 训练完成后，自动执行 LoRA **权重合并** (`merge_lora_weights.py`) 和**模型格式转换** (`convert_hf_to_gguf.py`)。

## 6. 输出

L2 层的最终输出是一个经过个性化微调的语言模型（可能是合并了 LoRA 权重的完整模型，或仅 LoRA 权重本身），通常会转换为 GGUF 等格式，以便在特定环境（如本地设备）高效运行。

## 7. 总结

L2 层是实现 "Second Me" 个性化的关键。它通过一个复杂的流程，将 L1 结构化的记忆和经 GraphRAG 增强的数据，转化为 SFT 和 DPO 所需的训练样本。通过 `trl` 等先进的对齐技术对基础 LLM 进行微调，最终产出一个能够深度理解和模拟用户的个性化模型。整个过程由 `TrainProcessService` 精确协调，从数据准备到模型训练、合并、转换，形成了一个完整的端到端个性化模型生产线。 