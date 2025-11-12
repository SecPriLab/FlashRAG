"""
构建 FAISS 索引脚本
使用 BGE 编码器为 index_corpus.jsonl 构建向量索引
"""

import os
import json
from flashrag.retriever.index_builder import DenseIndexBuilder
from flashrag.config import Config

def build_faiss_index():
    """
    使用 BGE 编码器构建 FAISS 索引
    """
    # 配置参数
    config_dict = {
        # 数据路径
        'corpus_path': '/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl',
        'save_dir': '/home/user/FlashRAG/indexes/scifact',

        # 检索配置
        'retrieval_method': 'bge',  # 使用 BGE 编码器
        'retrieval_model_path': 'BAAI/bge-large-en-v1.5',  # BGE 模型路径
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'retrieval_batch_size': 256,
        'faiss_gpu': True,  # 使用 GPU 加速

        # 索引配置
        'faiss_type': 'flat',  # 使用 Flat 索引（精确搜索）
        'embedding_dim': 1024,  # BGE-large 的维度

        # 其他配置
        'use_sentence_transformer': False,
        'instruction': '',
    }

    config = Config(config_dict=config_dict)

    # 检查语料库文件是否存在
    if not os.path.exists(config_dict['corpus_path']):
        print(f"❌ Error: Corpus file not found at {config_dict['corpus_path']}")
        print("Please run prepare_mia_data.py first to generate the corpus.")
        return

    # 创建保存目录
    os.makedirs(config_dict['save_dir'], exist_ok=True)

    print("🚀 Starting index building...")
    print(f"📁 Corpus path: {config_dict['corpus_path']}")
    print(f"💾 Save directory: {config_dict['save_dir']}")
    print(f"🔧 Retrieval method: {config_dict['retrieval_method']}")
    print(f"🤖 Model: {config_dict['retrieval_model_path']}")

    # 构建索引
    try:
        index_builder = DenseIndexBuilder(config)
        index_builder.build_index()
        print("\n✅ Index built successfully!")
        print(f"📍 Index saved to: {config_dict['save_dir']}")
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    build_faiss_index()
