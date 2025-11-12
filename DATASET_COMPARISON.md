# Dataset.py 文件对比分析报告

## 📋 概述

`dataset.py` 是 FlashRAG 框架中负责数据管理的核心模块。虽然从表面上看，官方 FlashRAG 和 RAG_MIA 的 `dataset.py` 结构相似，但可能存在一些针对 MIA 攻击需求的细微差异。

---

## 🔍 可能的差异点

### 1. **Output 字段的扩展**

#### 官方 FlashRAG 的 output 字段通常包含：
```python
item.output = {
    'pred': '...',              # 预测答案
    'prompt': '...',            # 使用的提示词
    'retrieval_result': [...],  # 检索结果
    'metric_score': {...}       # 评估分数
}
```

#### RAG_MIA 可能需要额外的字段：
```python
item.output = {
    'pred': '...',
    'prompt': '...',
    'retrieval_result': [...],

    # MIA 专用字段
    'yes_prob': 0.85,           # yes token 的概率
    'no_prob': 0.15,            # no token 的概率
    'perplexity': 12.34,        # 困惑度
    'is_member': True,          # 成员标记（ground truth）
    'is_perturbed': False,      # 是否是扰动样本
    'original_id': '1234',      # 原始样本ID（用于配对）
}
```

### 2. **Metadata 字段的差异**

#### 官方 FlashRAG 的 metadata：
```python
item.metadata = {
    'source': 'nq',
    'difficulty': 'hard',
    'category': 'science'
}
```

#### RAG_MIA 需要的 metadata：
```python
item.metadata = {
    'source': 'nq',
    'difficulty': 'hard',
    'category': 'science',

    # MIA 专用元数据
    'doc_id': '4983',           # 对应的文档ID
    'is_member': True,          # 是否是成员数据
    'perturbation_ratio': 0.03, # 扰动比例
    'original_text': '...',     # 原始文本（用于对比）
}
```

### 3. **数据加载逻辑可能的差异**

#### 标准加载（官方 FlashRAG）：
```python
def _load_data(self, dataset_name, dataset_path):
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            item_dict = json.loads(line)
            item = Item(item_dict)
            data.append(item)
    return data
```

#### MIA 可能需要的配对加载：
```python
def _load_data(self, dataset_name, dataset_path):
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            item_dict = json.loads(line)

            # MIA 特殊处理：标记成员/非成员
            if 'is_member' not in item_dict.get('metadata', {}):
                # 从文件名或其他方式推断
                if 'member' in dataset_path:
                    item_dict['metadata']['is_member'] = True
                elif 'nonmember' in dataset_path:
                    item_dict['metadata']['is_member'] = False

            item = Item(item_dict)
            data.append(item)

    return data
```

---

## 🎯 为什么 Dataset.py 会有差异？

### 原因 1：**数据标注需求**

MIA 攻击需要明确的成员/非成员标签：

```python
# 官方 FlashRAG：只关心 QA 任务
{
    "id": "1",
    "question": "What is...?",
    "golden_answers": ["Answer"]
}

# RAG_MIA：需要额外的成员标注
{
    "id": "1",
    "question": "What is...?",
    "golden_answers": ["Answer"],
    "metadata": {
        "is_member": true,      # ← 关键差异
        "doc_id": "4983"        # ← 对应的文档ID
    }
}
```

### 原因 2：**扰动样本配对**

MIA 需要维护原始样本和扰动样本的对应关系：

```python
# 原始样本
original_item = Item({
    "id": "member_0",
    "question": "original question",
    "metadata": {
        "is_member": True,
        "is_perturbed": False,
        "pair_id": "member_0"  # ← 配对标识
    }
})

# 扰动样本
perturbed_item = Item({
    "id": "member_0_perturbed",
    "question": "perturbed question",
    "metadata": {
        "is_member": True,
        "is_perturbed": True,
        "pair_id": "member_0",     # ← 指向原始样本
        "original_id": "member_0"  # ← 原始样本ID
    }
})
```

### 原因 3：**概率数据存储**

MIA 需要存储和管理概率数据：

```python
# Dataset 类可能新增方法
class Dataset:
    def save_probabilities(self, save_path: str):
        """保存所有样本的概率数据"""
        probs = []
        for item in self.data:
            if 'yes_prob' in item.output and 'no_prob' in item.output:
                probs.append({
                    'id': item.id,
                    'is_member': item.metadata.get('is_member'),
                    'yes_prob': item.output['yes_prob'],
                    'no_prob': item.output['no_prob'],
                    'is_perturbed': item.metadata.get('is_perturbed', False)
                })

        # 保存为 PyTorch tensor
        import torch
        member_probs = [p['yes_prob'] for p in probs if p['is_member'] and not p['is_perturbed']]
        torch.save(torch.tensor(member_probs), f"{save_path}/member_yes_probs.pt")

    def load_paired_dataset(self, original_path: str, perturbed_path: str):
        """加载原始和扰动样本的配对数据集"""
        original_data = self._load_data('original', original_path)
        perturbed_data = self._load_data('perturbed', perturbed_path)

        # 构建配对映射
        self.pairs = {}
        for orig, pert in zip(original_data, perturbed_data):
            self.pairs[orig.id] = {
                'original': orig,
                'perturbed': pert
            }
```

---

## 📊 具体可能存在的代码差异

### 差异 1：Item.update_output() 方法

#### 官方 FlashRAG（严格限制）：
```python
def update_output(self, key: str, value: Any) -> None:
    if key in ["id", "question", "golden_answers", "output", "choices"]:
        raise AttributeError(f"{key} should not be changed")
    else:
        self.output[key] = value
```

#### RAG_MIA 可能的修改（允许更新某些字段）：
```python
def update_output(self, key: str, value: Any) -> None:
    # 可能放宽限制，允许更新某些字段用于 MIA 分析
    protected_fields = ["id", "question", "golden_answers", "choices"]
    if key in protected_fields:
        raise AttributeError(f"{key} should not be changed")
    else:
        self.output[key] = value

    # 或者添加特殊处理
    if key in ['yes_prob', 'no_prob', 'perplexity']:
        # 自动转换为 float 或 tensor
        self.output[key] = float(value)
```

### 差异 2：Dataset.save() 方法

#### 官方 FlashRAG：
```python
def save(self, save_path: str) -> None:
    save_data = [item.to_dict() for item in self.data]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
```

#### RAG_MIA 可能的扩展：
```python
def save(self, save_path: str, save_tensors: bool = False) -> None:
    # 标准的 JSON 保存
    save_data = [item.to_dict() for item in self.data]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    # 额外保存概率张量（用于 MIA 分析）
    if save_tensors:
        import torch
        base_path = os.path.dirname(save_path)

        # 保存 yes/no 概率
        member_yes = [item.output.get('yes_prob', 0) for item in self.data
                     if item.metadata.get('is_member', False)
                     and not item.metadata.get('is_perturbed', False)]

        perturb_member_yes = [item.output.get('yes_prob', 0) for item in self.data
                             if item.metadata.get('is_member', False)
                             and item.metadata.get('is_perturbed', False)]

        torch.save(torch.tensor(member_yes), f"{base_path}/member_yes_probs.pt")
        torch.save(torch.tensor(perturb_member_yes), f"{base_path}/perturb_member_yes_probs.pt")
```

### 差异 3：新增的配对方法

RAG_MIA 可能添加了专门处理配对数据的方法：

```python
class Dataset:
    def split_by_member_status(self):
        """按成员/非成员状态分割数据集"""
        members = [item for item in self.data if item.metadata.get('is_member', False)]
        nonmembers = [item for item in self.data if not item.metadata.get('is_member', True)]

        return Dataset(data=members), Dataset(data=nonmembers)

    def split_by_perturbation(self):
        """按是否扰动分割数据集"""
        original = [item for item in self.data if not item.metadata.get('is_perturbed', False)]
        perturbed = [item for item in self.data if item.metadata.get('is_perturbed', False)]

        return Dataset(data=original), Dataset(data=perturbed)

    def get_paired_items(self):
        """获取配对的原始和扰动样本"""
        pairs = []

        original_items = {item.id: item for item in self.data
                         if not item.metadata.get('is_perturbed', False)}
        perturbed_items = {item.metadata.get('original_id'): item for item in self.data
                          if item.metadata.get('is_perturbed', False)}

        for orig_id in original_items:
            if orig_id in perturbed_items:
                pairs.append({
                    'original': original_items[orig_id],
                    'perturbed': perturbed_items[orig_id]
                })

        return pairs
```

---

## 🔬 实际使用场景对比

### 场景 1：标准 RAG 任务（官方 FlashRAG）

```python
# 加载数据集
dataset = Dataset(
    config=config,
    dataset_path='data/nq/test.jsonl'
)

# 运行 pipeline
pipeline = SequentialPipeline(config)
result = pipeline.run(dataset)

# 评估
evaluator.evaluate(dataset)

# 保存结果
dataset.save('output/results.json')
```

### 场景 2：MIA 攻击任务（RAG_MIA）

```python
# 1. 加载成员和非成员数据
member_dataset = Dataset(
    config=config,
    dataset_path='data/member_samples.jsonl'
)

nonmember_dataset = Dataset(
    config=config,
    dataset_path='data/nonmember_samples.jsonl'
)

# 2. 运行 RAG pipeline（收集概率）
pipeline = SequentialPipeline(config)
member_result = pipeline.run(member_dataset)      # 自动保存 yes/no 概率
nonmember_result = pipeline.run(nonmember_dataset)

# 3. 加载扰动样本
perturb_member_dataset = Dataset(
    config=config,
    dataset_path='data/perturb_member_samples.jsonl'
)

perturb_nonmember_dataset = Dataset(
    config=config,
    dataset_path='data/perturb_nonmember_samples.jsonl'
)

# 4. 再次运行 pipeline
perturb_member_result = pipeline.run(perturb_member_dataset)
perturb_nonmember_result = pipeline.run(perturb_nonmember_dataset)

# 5. 提取概率数据（Dataset 可能有专门的方法）
member_dataset.save_probabilities('output/member')
perturb_member_dataset.save_probabilities('output/perturb_member')
nonmember_dataset.save_probabilities('output/nonmember')
perturb_nonmember_dataset.save_probabilities('output/perturb_nonmember')

# 6. 执行 MIA 攻击（使用保存的概率）
# 这部分在 MIA.py 中完成
```

---

## 💡 为什么这些差异是必要的？

### 1. **数据组织需求**

MIA 攻击需要严格的数据组织：
- ✅ 成员/非成员明确分离
- ✅ 原始/扰动样本配对
- ✅ 概率数据与样本关联

### 2. **实验重现性**

MIA 研究需要保存完整的实验数据：
- ✅ 保存原始输入
- ✅ 保存模型输出（概率）
- ✅ 保存元数据（成员标签、扰动信息）

### 3. **分析便利性**

MIA 分析需要高效的数据访问：
- ✅ 快速分割数据集（按成员状态、扰动状态）
- ✅ 快速提取概率数据
- ✅ 自动保存为 PyTorch 张量格式

---

## 🎯 对你的项目的启示

你当前的实现已经很完善，但如果要进一步支持 MIA 攻击，可以考虑扩展 Dataset 类：

### 建议 1：添加 MIA 专用的数据管理方法

```python
# 在你的项目中添加一个 MIA 数据集类
class MIADataset(Dataset):
    """扩展 Dataset 类以支持 MIA 攻击"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_mia_data()

    def _validate_mia_data(self):
        """验证数据集是否包含 MIA 所需的字段"""
        for item in self.data:
            if 'is_member' not in item.metadata:
                raise ValueError(f"Item {item.id} missing 'is_member' in metadata")

    def save_probabilities_as_tensors(self, save_dir: str):
        """保存概率数据为 PyTorch 张量"""
        import torch
        import os

        os.makedirs(save_dir, exist_ok=True)

        # 提取数据
        member_probs = []
        nonmember_probs = []

        for item in self.data:
            if 'answer_probs' not in item.output:
                continue

            # 假设使用第一个答案的概率作为置信度
            prob = list(item.output['answer_probs'].values())[0]

            if item.metadata.get('is_member'):
                member_probs.append(prob)
            else:
                nonmember_probs.append(prob)

        # 保存
        if member_probs:
            torch.save(torch.tensor(member_probs),
                      f"{save_dir}/member_probs.pt")
        if nonmember_probs:
            torch.save(torch.tensor(nonmember_probs),
                      f"{save_dir}/nonmember_probs.pt")

    def get_paired_samples(self):
        """获取原始-扰动样本对"""
        pairs = {}

        for item in self.data:
            if item.metadata.get('is_perturbed'):
                orig_id = item.metadata.get('original_id')
                if orig_id not in pairs:
                    pairs[orig_id] = {}
                pairs[orig_id]['perturbed'] = item
            else:
                if item.id not in pairs:
                    pairs[item.id] = {}
                pairs[item.id]['original'] = item

        return pairs
```

### 建议 2：修改数据准备脚本

```python
# 在 prepare_mia_data.py 中确保数据格式正确
def create_query_dataset(member_docs, nonmember_docs, output_path):
    queries = []

    # 成员样本
    for idx, doc in enumerate(member_docs):
        query = {
            'id': f'member_{idx}',
            'question': doc['title'],
            'golden_answers': [],
            'metadata': {
                'doc_id': doc['_id'],
                'is_member': True,           # ← MIA 必需
                'is_perturbed': False,       # ← MIA 必需
                'full_text': doc['text']
            }
        }
        queries.append(query)

    # 非成员样本
    for idx, doc in enumerate(nonmember_docs):
        query = {
            'id': f'nonmember_{idx}',
            'question': doc['title'],
            'golden_answers': [],
            'metadata': {
                'doc_id': doc['_id'],
                'is_member': False,          # ← MIA 必需
                'is_perturbed': False,       # ← MIA 必需
                'full_text': doc['text']
            }
        }
        queries.append(query)

    save_jsonl(queries, output_path)
```

---

## 📖 总结

### Dataset.py 可能存在的差异原因：

1. **数据标注**：MIA 需要明确的成员/非成员标签
2. **配对管理**：MIA 需要维护原始-扰动样本的对应关系
3. **概率存储**：MIA 需要高效地保存和加载概率数据
4. **分析便利**：MIA 需要专门的数据分割和提取方法

### 核心差异点：

| 方面 | 官方 FlashRAG | RAG_MIA 可能的修改 |
|------|--------------|-------------------|
| **Metadata** | 基本任务信息 | 添加 is_member, is_perturbed 等 |
| **Output** | 预测结果 | 添加 yes_prob, no_prob, perplexity 等 |
| **Save** | 保存 JSON | 额外保存 PyTorch 张量 |
| **加载** | 标准加载 | 可能支持配对加载 |
| **新方法** | 无 | split_by_member_status(), save_probabilities() 等 |

### 对你的项目：

你的当前实现已经支持了 MIA 的核心功能（返回 logits、文档ID等），如果需要完整的 MIA 攻击流程，建议：

1. ✅ 扩展 Dataset 类添加 MIA 专用方法
2. ✅ 在数据准备阶段添加必需的 metadata 字段
3. ✅ 实现概率数据的批量保存和加载
4. ✅ 添加配对样本管理功能

---

**报告完成时间**: 2025-11-12
