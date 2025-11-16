# ConfidenceMIA - Membership Inference Attack for RAG Systems

## 概述

ConfidenceMIA是一个基于FlashRAG的文档成员推理类，用于推断某个成员文档是否出现在基于RAG的LLM的文档库中。

## 原理

该方法通过分析LLM在多轮挑战对话中的置信度变化来判断文档是否为成员文档：

1. **初始轮次**：提出针对目标文档的特定单项选择题Q
   - 如果目标文档是成员文档且LLM检索到该文档，会给出正确答案
   - 但有时LLM通过瞎猜或自身知识也能正确回答，导致误判

2. **挑战轮次**：提出挑战提示词C，认为LLM的回答不对，应该选择另一个不正确的选项
   - 如果LLM回答正确是因为检索到相关文档作为支撑，接下来回答正确答案的概率依然很高
   - 如果LLM回答正确是因为瞎猜或自身知识，接下来回答正确答案的概率将会急剧降低

3. **判断**：通过捕获LLM在回答过程中特征的时序变化来判断当前文档是成员还是非成员

## 功能特性

- ✅ 支持批量处理多个文档
- ✅ 支持多轮挑战对话
- ✅ 支持固定或随机挑战选项
- ✅ 自动保存responses和logits到HDF5文件
- ✅ 完整的元数据记录

## 安装

确保已安装FlashRAG及其依赖：

```bash
pip install flashrag
pip install h5py
```

## 使用方法

### 1. 准备配置

```python
from flashrag.mia import ConfidenceMIA

config = {
    # 检索器设置
    "retrieval_method": "bge",
    "retrieval_model_path": "/path/to/bge-model",
    "index_path": "/path/to/faiss-index",
    "corpus_path": "/path/to/corpus.jsonl",
    "retrieval_topk": 5,
    "faiss_gpu": True,

    # 生成器设置
    "framework": "hf",
    "generator_model": "llama3-8B-instruct",
    "generator_model_path": "/path/to/llama-model",
    "generator_max_input_len": 1024,
    "generator_batch_size": 4,
    "generation_params": {
        "do_sample": False,
        "max_tokens": 1
    },
    "device": "cuda:0",
    "gpu_num": 1,

    # MIA设置
    "num_challenges": 3,  # 挑战次数
    "random_challenge": False,  # 是否随机挑战
    "mia_data_path": "/path/to/data.jsonl",
    "mia_output_path": "mia_results.h5"
}
```

### 2. 运行攻击

```python
# 初始化ConfidenceMIA
mia = ConfidenceMIA(config)

# 执行攻击
mia.attack()
```

### 3. 读取结果

```python
import h5py
import json

with h5py.File('mia_results.h5', 'r') as h5f:
    for doc_id in h5f.keys():
        doc_group = h5f[doc_id]

        # 获取元数据
        is_member = doc_group.attrs['mem']
        num_questions = doc_group.attrs['num_questions']
        answers = json.loads(doc_group.attrs['answers'])

        # 获取responses和logits
        responses = doc_group['responses'][:]  # [num_questions, num_rounds]
        logits = doc_group['logits'][:]  # [num_questions, num_rounds, max_tokens, vocab_size]

        print(f"Document {doc_id}: Member={is_member}")
        print(f"  Responses shape: {responses.shape}")
        print(f"  Logits shape: {logits.shape}")
```

## 数据格式

### 输入数据格式（JSONL）

```json
{
    "id": "18264714",
    "text": "文档全文...",
    "sorted_sentences": ["句子1", "句子2", ...],
    "sorted_self_infos": [3.78, 3.47, ...],
    "len_sentences": 7,
    "questions": ["问题1", "问题2", ...],
    "answers": ["C", "C", "C", ...],
    "mem": "No"
}
```

### 输出数据格式（HDF5）

```
mia_results.h5
├── doc_id_1/
│   ├── responses (dataset): [num_questions, num_rounds] string array
│   ├── logits (dataset): [num_questions, num_rounds, max_tokens, vocab_size] float32 array
│   └── attributes:
│       ├── mem: "Yes" or "No"
│       ├── num_questions: int
│       ├── answers: JSON string of answer list
│       └── num_rounds: int
├── doc_id_2/
│   └── ...
```

## 配置参数说明

### MIA特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_challenges` | int | 3 | 挑战次数（总对话轮数 = num_challenges + 1） |
| `random_challenge` | bool | False | 是否随机挑战选项。False=固定错误选项，True=每轮随机 |
| `mia_data_path` | str | - | 输入数据的JSONL文件路径 |
| `mia_output_path` | str | "mia_results.h5" | 输出HDF5文件路径 |

### FlashRAG参数

参考FlashRAG官方文档配置retriever和generator相关参数。

## 示例

完整示例请参考 `example_usage.py`：

```bash
python flashrag/mia/example_usage.py
```

## 方法流程

1. **加载数据**：读取包含问题和答案的JSONL文件
2. **批量检索**：对每个文档的所有问题进行批量检索
3. **构建消息**：为每个问题构建包含检索上下文的system prompt
4. **多轮对话**：
   - 第一轮：提出选择题，获取初始回答
   - 后续轮：添加挑战提示词，观察回答变化
5. **保存结果**：将所有轮次的responses和logits保存到HDF5文件

## 注意事项

1. **内存使用**：处理大量文档时注意GPU内存使用
2. **批处理**：合理设置`generator_batch_size`以平衡速度和内存
3. **挑战模式**：
   - `random_challenge=False`：每个问题在所有轮次使用同一个错误选项（推荐用于一致性分析）
   - `random_challenge=True`：每轮随机选择错误选项（推荐用于鲁棒性测试）
4. **max_tokens**：建议设置为1，因为只需要单个字母的回答（A/B/C/D/E）

## 输出分析

生成的logits可用于：
- 计算每个选项的概率分布
- 分析置信度随挑战轮次的变化趋势
- 训练分类器区分成员/非成员文档

## 引用

如果使用此代码，请引用FlashRAG：

```bibtex
@article{flashrag,
  title={FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

遵循FlashRAG的许可证。
