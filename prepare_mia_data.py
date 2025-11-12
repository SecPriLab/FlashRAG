"""
数据准备脚本：用于成员推理攻击实验
1. 从 corpus.jsonl 随机采样成员和非成员样本
2. 创建索引语料库（排除非成员样本）
3. 生成查询数据集
"""

import json
import random
import os
from pathlib import Path

def load_corpus(corpus_path):
    """加载 corpus.jsonl 文件"""
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus.append(doc)
    print(f"Loaded {len(corpus)} documents from corpus")
    return corpus

def sample_member_nonmember(corpus, member_size=1000, nonmember_size=1000, seed=42):
    """
    随机采样成员和非成员样本

    Args:
        corpus: 完整的文档列表
        member_size: 成员样本数量
        nonmember_size: 非成员样本数量
        seed: 随机种子

    Returns:
        member_docs: 成员样本列表
        nonmember_docs: 非成员样本列表
        index_corpus: 用于建立索引的语料库（排除非成员）
    """
    random.seed(seed)

    # 确保有足够的文档
    total_needed = member_size + nonmember_size
    if len(corpus) < total_needed:
        raise ValueError(f"Corpus has only {len(corpus)} docs, but need {total_needed}")

    # 随机打乱
    shuffled_corpus = corpus.copy()
    random.shuffle(shuffled_corpus)

    # 采样
    member_docs = shuffled_corpus[:member_size]
    nonmember_docs = shuffled_corpus[member_size:member_size + nonmember_size]

    # 创建非成员ID集合，用于快速查找
    nonmember_ids = set(doc['_id'] for doc in nonmember_docs)

    # 创建索引语料库（排除非成员）
    index_corpus = [doc for doc in corpus if doc['_id'] not in nonmember_ids]

    print(f"Sampled {len(member_docs)} member documents")
    print(f"Sampled {len(nonmember_docs)} non-member documents")
    print(f"Index corpus contains {len(index_corpus)} documents")

    return member_docs, nonmember_docs, index_corpus

def save_jsonl(data, output_path):
    """保存为 jsonl 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {output_path}")

def create_query_dataset(member_docs, nonmember_docs, output_path):
    """
    创建查询数据集，格式符合 FlashRAG 要求

    格式：
    {
        'id': str,
        'question': str,  # 使用 title 作为查询
        'golden_answers': [],  # 空列表，因为不需要评估
        'metadata': {
            'doc_id': str,  # 原始文档ID
            'is_member': bool,  # 是否为成员
            'full_text': str  # 完整文本，用于后续分析
        }
    }
    """
    queries = []

    # 处理成员样本
    for idx, doc in enumerate(member_docs):
        query = {
            'id': f'member_{idx}',
            'question': doc['title'],  # 使用标题作为查询
            'golden_answers': [],
            'metadata': {
                'doc_id': doc['_id'],
                'is_member': True,
                'full_text': doc['text']
            }
        }
        queries.append(query)

    # 处理非成员样本
    for idx, doc in enumerate(nonmember_docs):
        query = {
            'id': f'nonmember_{idx}',
            'question': doc['title'],  # 使用标题作为查询
            'golden_answers': [],
            'metadata': {
                'doc_id': doc['_id'],
                'is_member': False,
                'full_text': doc['text']
            }
        }
        queries.append(query)

    # 保存查询数据集
    save_jsonl(queries, output_path)

    return queries

def main():
    # 配置路径
    corpus_path = '/home/user/FlashRAG/datasets/scifact/corpus.jsonl'
    output_dir = Path('/home/user/FlashRAG/datasets/scifact')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 corpus.jsonl 是否存在
    if not os.path.exists(corpus_path):
        print(f"❌ Error: {corpus_path} does not exist!")
        print("Please place your corpus.jsonl file in /home/user/FlashRAG/datasets/scifact/")
        return

    # 1. 加载语料库
    corpus = load_corpus(corpus_path)

    # 2. 采样成员和非成员
    member_docs, nonmember_docs, index_corpus = sample_member_nonmember(
        corpus,
        member_size=1000,
        nonmember_size=1000,
        seed=42
    )

    # 3. 保存采样结果
    save_jsonl(member_docs, output_dir / 'member_samples.jsonl')
    save_jsonl(nonmember_docs, output_dir / 'nonmember_samples.jsonl')
    save_jsonl(index_corpus, output_dir / 'index_corpus.jsonl')

    # 4. 创建查询数据集
    queries = create_query_dataset(member_docs, nonmember_docs, output_dir / 'queries.jsonl')

    print("\n✅ Data preparation completed!")
    print(f"📁 Output files:")
    print(f"  - member_samples.jsonl: {len(member_docs)} documents")
    print(f"  - nonmember_samples.jsonl: {len(nonmember_docs)} documents")
    print(f"  - index_corpus.jsonl: {len(index_corpus)} documents")
    print(f"  - queries.jsonl: {len(queries)} queries")

if __name__ == '__main__':
    main()
