# FlashRAG 成员推理攻击 (MIA) 实验指南

本指南介绍如何使用修改后的 FlashRAG 进行成员推理攻击实验。

## 📋 目录

1. [功能概述](#功能概述)
2. [安装要求](#安装要求)
3. [快速开始](#快速开始)
4. [详细使用说明](#详细使用说明)
5. [API 参考](#api-参考)
6. [常见问题](#常见问题)

---

## 🎯 功能概述

本项目对 FlashRAG 进行了以下扩展，以支持成员推理攻击实验：

### 1. **数据准备**
- ✅ 从 corpus.jsonl 随机采样成员和非成员样本
- ✅ 创建用于索引的语料库（排除非成员样本）
- ✅ 生成查询数据集

### 2. **检索增强**
- ✅ 使用 BGE 编码器构建 FAISS 索引
- ✅ **新增**：检索器返回文档 ID
- ✅ 支持 GPU 加速

### 3. **生成增强**
- ✅ 使用 Llama-3.1-8B-Instruct 生成答案
- ✅ **新增**：返回生成的 logits
- ✅ 提取特定答案（A-E）的概率

### 4. **多轮对话**
- ✅ 支持选择是否检索
- ✅ 维护对话历史
- ✅ 返回详细的推理信息

### 5. **提示词定制**
- ✅ 新增 MIA 专用提示词模板
- ✅ 支持自定义系统提示词

---

## 📦 安装要求

### 前置条件
```bash
# 已安装的包
- Python >= 3.8
- PyTorch >= 2.0
- CUDA (for GPU support)
- FlashRAG (pip install -e .)
- faiss-gpu
```

### 验证安装
```python
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline

config = Config()
print("FlashRAG installation successful!")
```

---

## 🚀 快速开始

### 步骤 1：准备数据

将您的 `corpus.jsonl` 文件放置在以下位置：
```
FlashRAG/datasets/scifact/corpus.jsonl
```

corpus.jsonl 格式：
```json
{"_id": "4983", "title": "...", "text": "...", "metadata": {}}
{"_id": "5836", "title": "...", "text": "...", "metadata": {}}
```

运行数据准备脚本：
```bash
cd /home/user/FlashRAG
python prepare_mia_data.py
```

输出文件：
- `datasets/scifact/member_samples.jsonl` - 1000个成员样本
- `datasets/scifact/nonmember_samples.jsonl` - 1000个非成员样本
- `datasets/scifact/index_corpus.jsonl` - 索引语料库
- `datasets/scifact/queries.jsonl` - 查询数据集

### 步骤 2：构建索引

```bash
python build_index.py
```

这将使用 BGE 编码器为 `index_corpus.jsonl` 构建 FAISS 索引。

输出：
- `indexes/scifact/index` - FAISS 索引文件

### 步骤 3：测试功能

```bash
python test_mia_pipeline.py
```

这将运行所有测试，验证：
- 检索器返回文档 ID
- 生成器返回 logits
- 多轮对话功能
- 完整的 pipeline

---

## 📖 详细使用说明

### 1. 数据准备详解

`prepare_mia_data.py` 脚本执行以下操作：

```python
# 主要功能
def sample_member_nonmember(corpus, member_size=1000, nonmember_size=1000, seed=42):
    """
    Args:
        corpus: 完整的文档列表
        member_size: 成员样本数量 (默认 1000)
        nonmember_size: 非成员样本数量 (默认 1000)
        seed: 随机种子 (默认 42)

    Returns:
        member_docs: 成员样本列表
        nonmember_docs: 非成员样本列表
        index_corpus: 用于建立索引的语料库（排除非成员）
    """
```

**自定义采样大小：**
```python
# 修改 prepare_mia_data.py 中的参数
member_docs, nonmember_docs, index_corpus = sample_member_nonmember(
    corpus,
    member_size=500,  # 改为 500
    nonmember_size=500,  # 改为 500
    seed=42
)
```

### 2. 索引构建详解

`build_index.py` 使用以下配置：

```python
config_dict = {
    'corpus_path': '/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl',
    'save_dir': '/home/user/FlashRAG/indexes/scifact',
    'retrieval_method': 'bge',  # BGE 编码器
    'retrieval_model_path': 'BAAI/bge-large-en-v1.5',
    'faiss_type': 'flat',  # 精确搜索
    'faiss_gpu': True,  # GPU 加速
}
```

**更改编码器模型：**
```python
# 在 build_index.py 中修改
'retrieval_model_path': 'BAAI/bge-base-en-v1.5',  # 使用 base 模型
'embedding_dim': 768,  # base 模型的维度
```

### 3. 多轮对话详解

#### 基本使用

```python
from mia_multi_turn_chat import create_mia_chat

# 创建对话实例
chat = create_mia_chat(
    model_path="/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct",
    retrieval_method="bge",
    corpus_path="/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl",
    index_path="/home/user/FlashRAG/indexes/scifact",
    retrieval_topk=3
)

# 带检索的对话
result = chat.chat(
    user_query="What is the role of MDSC in myelodysplasia?",
    use_retrieval=True,
    topk=3,
    return_details=True
)

# 访问结果
print(f"Response: {result['response']}")
print(f"Retrieved Doc IDs: {result['retrieved_doc_ids']}")
print(f"Answer Probabilities: {result['answer_probs']}")
print(f"Predicted Answer: {result['predicted_answer']}")
```

#### 返回值详解

```python
result = {
    'user_query': str,              # 用户查询
    'use_retrieval': bool,          # 是否使用检索
    'response': str,                # 生成的回答
    'retrieved_doc_ids': List[str], # 检索到的文档ID（如果使用检索）
    'retrieved_documents': List[Dict],  # 检索到的完整文档
    'logits': torch.Tensor,         # 第一个token的logits [vocab_size]
    'answer_probs': Dict[str, float],  # A-E的概率 {'A': 0.1, 'B': 0.2, ...}
    'predicted_answer': str         # 预测的答案字母
}
```

#### 不带检索的对话

```python
# 不使用检索，直接生成
result = chat.chat(
    user_query="Can you explain more?",
    use_retrieval=False
)
```

#### 对话历史管理

```python
# 查看对话历史
history = chat.get_conversation_history()

# 打印对话历史
chat.print_conversation_history()

# 重置对话
chat.reset_conversation()
```

### 4. 单次查询的完整 Pipeline

```python
from flashrag.config import Config
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate

# 配置
config_dict = {
    'generator_model_path': '/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct',
    'retrieval_method': 'bge',
    'corpus_path': '/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl',
    'index_path': '/home/user/FlashRAG/indexes/scifact',
    'retrieval_topk': 3,
    'faiss_gpu': True,
}

config = Config(config_dict=config_dict)

# 初始化组件
retriever = get_retriever(config)
generator = get_generator(config)
prompt_template = PromptTemplate(
    config,
    system_prompt=PromptTemplate.mia_system_prompt
)

# 查询
query = "Your question here"

# 1. 检索（返回文档ID）
results, scores, doc_ids = retriever._batch_search(
    query=[query],
    num=3,
    return_score=True,
    return_doc_ids=True  # 新增参数
)

print(f"Retrieved document IDs: {doc_ids[0]}")

# 2. 构建 prompt
input_prompt = prompt_template.get_string(
    question=query,
    retrieval_result=results[0]
)

# 3. 生成（返回logits）
output = generator.generate(
    [input_prompt],
    return_dict=True,  # 返回详细信息
    max_new_tokens=10
)

response = output['responses'][0]
logits = output['generated_token_logits'][0]  # [num_tokens, vocab_size]

# 4. 提取答案概率
answer_tokens = ['A', 'B', 'C', 'D', 'E']
answer_token_ids = {
    token: generator.tokenizer.convert_tokens_to_ids(token)
    for token in answer_tokens
}

first_token_logits = logits[0]  # 第一个token的logits
answer_probs = {
    token: first_token_logits[token_id].item()
    for token, token_id in answer_token_ids.items()
}

print(f"Response: {response}")
print(f"Answer Probabilities: {answer_probs}")
```

---

## 📚 API 参考

### MIAMultiTurnChat 类

#### 初始化

```python
from mia_multi_turn_chat import create_mia_chat

chat = create_mia_chat(
    model_path: str,           # LLM 模型路径
    retrieval_method: str,     # 检索方法 (默认 'bge')
    corpus_path: str,          # 语料库路径
    index_path: str,           # 索引路径
    retrieval_topk: int        # 检索文档数量 (默认 3)
)
```

#### 方法

##### `chat(user_query, use_retrieval=True, topk=3, return_details=True)`

执行一轮对话。

**参数：**
- `user_query` (str): 用户查询
- `use_retrieval` (bool): 是否使用检索（默认 True）
- `topk` (int): 检索文档数量（默认 3）
- `return_details` (bool): 是否返回详细信息（默认 True）

**返回：**
- dict: 包含 response, retrieved_doc_ids, logits, answer_probs 等

##### `reset_conversation()`

重置对话历史。

##### `get_conversation_history()`

获取对话历史列表。

**返回：**
- List[Dict]: 对话历史

##### `print_conversation_history()`

打印格式化的对话历史。

---

### 修改的 FlashRAG 组件

#### DenseRetriever

**新增参数：**

```python
retriever._batch_search(
    query: List[str],
    num: int = None,
    return_score: bool = False,
    return_doc_ids: bool = False  # 新增：返回文档ID
)
```

**返回值：**
- 如果 `return_doc_ids=True` 且 `return_score=True`：
  - `(results, scores, doc_ids)` 三元组
- 如果只有 `return_doc_ids=True`：
  - `(results, doc_ids)` 二元组

#### HFCausalLMGenerator

**已有功能（利用现有参数）：**

```python
generator.generate(
    input_list: List[str],
    return_dict: bool = True,  # 返回详细信息
    max_new_tokens: int = 10
)
```

**返回值（当 return_dict=True）：**
```python
{
    'responses': List[str],                    # 生成的文本
    'generated_token_ids': torch.Tensor,       # 生成的token IDs
    'generated_token_logits': torch.Tensor,    # 每个token的logits
    'scores': List[List[float]]                # 概率分数
}
```

#### PromptTemplate

**新增属性：**

```python
PromptTemplate.mia_system_prompt = (
    "You are a helpful assistant, below is a query from a user and some relevant documents."
    "Answer the question given the information in those documents."
    " Your answer should be short and concise. You can only output one letter from A, B, C, D, or E as your answer."
    "\nThe following are given documents.\n\n{reference}"
)
```

**使用方法：**

```python
prompt_template = PromptTemplate(
    config,
    system_prompt=PromptTemplate.mia_system_prompt,
    user_prompt=PromptTemplate.base_user_prompt
)
```

---

## ❓ 常见问题

### Q1: corpus.jsonl 的格式要求是什么？

A: 每行一个 JSON 对象，必须包含 `_id`, `title`, `text` 字段：

```json
{"_id": "4983", "title": "Document title", "text": "Document content", "metadata": {}}
```

### Q2: 如何更改成员/非成员样本的数量？

A: 修改 `prepare_mia_data.py` 中的 `sample_member_nonmember` 调用：

```python
member_docs, nonmember_docs, index_corpus = sample_member_nonmember(
    corpus,
    member_size=500,   # 改为所需数量
    nonmember_size=500,
    seed=42
)
```

### Q3: 如何使用不同的 LLM 模型？

A: 修改配置中的 `generator_model_path`：

```python
config_dict = {
    'generator_model_path': '/path/to/your/model',
    'generator_model': 'your-model-name',
}
```

### Q4: 如何使用不同的检索器？

A: 修改配置中的 `retrieval_method` 和 `retrieval_model_path`：

```python
config_dict = {
    'retrieval_method': 'e5',  # 或其他方法
    'retrieval_model_path': 'path/to/e5/model',
}
```

### Q5: 生成的 logits 是什么？

A: `generated_token_logits` 是模型为每个生成的 token 计算的未归一化分数（logits），形状为 `[num_generated_tokens, vocab_size]`。第一个 token 的 logits (`logits[0]`) 对应于答案选项（A-E）的原始分数。

### Q6: 如何提取正确答案的概率？

A: 参考以下代码：

```python
# 获取答案 token IDs
answer_token_ids = {
    token: generator.tokenizer.convert_tokens_to_ids(token)
    for token in ['A', 'B', 'C', 'D', 'E']
}

# 获取第一个token的logits
first_token_logits = output['generated_token_logits'][0][0]

# 提取每个答案的概率（logits已经过softmax）
answer_probs = {
    token: first_token_logits[token_id].item()
    for token, token_id in answer_token_ids.items()
}
```

### Q7: 多轮对话中的历史如何管理？

A: `MIAMultiTurnChat` 类自动维护历史。每次调用 `chat()` 都会：
1. 将用户查询添加到 `self.messages`
2. 将助手回复添加到 `self.messages`
3. 如果使用检索，还会保存检索到的文档ID

可以通过 `get_conversation_history()` 查看或 `reset_conversation()` 重置。

### Q8: 如何批量处理多个查询？

A: 使用循环或批处理：

```python
queries = [...]  # 查询列表

results = []
for query in queries:
    result = chat.chat(query, use_retrieval=True)
    results.append(result)

    # 每个查询后重置对话（如果需要独立处理）
    # chat.reset_conversation()
```

### Q9: GPU 内存不足怎么办？

A: 可以：
1. 减小 `generator_batch_size`
2. 减小 `retrieval_batch_size`
3. 使用 `retrieval_use_fp16=True`
4. 使用更小的模型

### Q10: 如何保存实验结果？

A: 将结果保存为 JSON：

```python
import json

results = []
for query in queries:
    result = chat.chat(query, use_retrieval=True)
    # 转换 tensor 为 list
    result['logits'] = result['logits'].tolist() if result['logits'] is not None else None
    results.append(result)

with open('mia_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

---

## 📝 完整示例脚本

```python
#!/usr/bin/env python
"""
完整的 MIA 实验示例
"""

from mia_multi_turn_chat import create_mia_chat
import json

# 1. 创建对话实例
chat = create_mia_chat()

# 2. 加载查询数据集
with open('/home/user/FlashRAG/datasets/scifact/queries.jsonl', 'r') as f:
    queries = [json.loads(line) for line in f]

# 3. 对成员和非成员样本进行推理
results = []

for query_item in queries[:10]:  # 处理前10个
    query_text = query_item['question']
    is_member = query_item['metadata']['is_member']
    doc_id = query_item['metadata']['doc_id']

    # 执行推理
    result = chat.chat(
        user_query=query_text,
        use_retrieval=True,
        topk=3
    )

    # 保存结果
    result_record = {
        'doc_id': doc_id,
        'is_member': is_member,
        'query': query_text,
        'response': result['response'],
        'retrieved_doc_ids': result['retrieved_doc_ids'],
        'answer_probs': result['answer_probs'],
        'predicted_answer': result['predicted_answer']
    }

    results.append(result_record)

    # 重置对话历史（每个查询独立）
    chat.reset_conversation()

# 4. 保存结果
with open('mia_experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Processed {len(results)} queries")
print(f"💾 Results saved to mia_experiment_results.json")
```

---

## 🔬 成员推理攻击分析

使用检索到的文档 ID 和答案概率进行成员推理攻击分析：

```python
import numpy as np

def analyze_membership_inference(results):
    """
    分析成员推理攻击效果

    Args:
        results: 包含 is_member, retrieved_doc_ids, answer_probs 的结果列表
    """
    member_scores = []
    nonmember_scores = []

    for result in results:
        doc_id = result['doc_id']
        is_member = result['is_member']
        retrieved_ids = result['retrieved_doc_ids']

        # 成员推理信号：文档是否在检索结果中
        is_retrieved = doc_id in retrieved_ids

        # 获取检索位置（越靠前，分数越高）
        if is_retrieved:
            position = retrieved_ids.index(doc_id)
            score = 1.0 / (position + 1)  # 位置越靠前，分数越高
        else:
            score = 0.0

        if is_member:
            member_scores.append(score)
        else:
            nonmember_scores.append(score)

    # 统计
    print(f"成员样本平均分数: {np.mean(member_scores):.4f}")
    print(f"非成员样本平均分数: {np.mean(nonmember_scores):.4f}")
    print(f"分数差异: {np.mean(member_scores) - np.mean(nonmember_scores):.4f}")

    # 计算 AUC
    from sklearn.metrics import roc_auc_score

    all_scores = member_scores + nonmember_scores
    all_labels = [1] * len(member_scores) + [0] * len(nonmember_scores)

    auc = roc_auc_score(all_labels, all_scores)
    print(f"AUC: {auc:.4f}")

    return auc

# 使用
auc = analyze_membership_inference(results)
```

---

## 📞 联系与支持

如有问题，请：
1. 检查本 README 的常见问题部分
2. 查看 FlashRAG 官方文档
3. 运行 `test_mia_pipeline.py` 进行调试

---

## 📄 许可证

遵循 FlashRAG 的原始许可证。

---

**Happy Experimenting! 🚀**
