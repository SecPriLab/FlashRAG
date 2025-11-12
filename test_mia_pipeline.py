"""
完整的 MIA Pipeline 测试脚本
测试所有功能：
1. 数据准备
2. 索引构建
3. 检索功能（返回文档ID）
4. 生成功能（返回logits）
5. 多轮对话
"""

import os
import json
import torch
from flashrag.config import Config
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate


def test_data_preparation():
    """测试数据准备功能"""
    print("\n" + "="*70)
    print("🧪 Test 1: Data Preparation")
    print("="*70)

    corpus_path = '/home/user/FlashRAG/datasets/scifact/corpus.jsonl'

    if not os.path.exists(corpus_path):
        print("❌ corpus.jsonl not found. Please place it in datasets/scifact/")
        return False

    # 运行数据准备脚本
    print("Running prepare_mia_data.py...")
    os.system('cd /home/user/FlashRAG && python prepare_mia_data.py')

    # 检查输出文件
    expected_files = [
        'datasets/scifact/member_samples.jsonl',
        'datasets/scifact/nonmember_samples.jsonl',
        'datasets/scifact/index_corpus.jsonl',
        'datasets/scifact/queries.jsonl'
    ]

    all_exist = True
    for file in expected_files:
        full_path = f'/home/user/FlashRAG/{file}'
        if os.path.exists(full_path):
            print(f"✅ {file} created")
        else:
            print(f"❌ {file} NOT found")
            all_exist = False

    return all_exist


def test_index_building():
    """测试索引构建"""
    print("\n" + "="*70)
    print("🧪 Test 2: Index Building")
    print("="*70)

    # 运行索引构建脚本
    print("Running build_index.py...")
    os.system('cd /home/user/FlashRAG && python build_index.py')

    # 检查索引文件
    index_path = '/home/user/FlashRAG/indexes/scifact/index'
    if os.path.exists(index_path):
        print(f"✅ Index file created at {index_path}")
        return True
    else:
        print(f"❌ Index file NOT found at {index_path}")
        return False


def test_retriever_with_doc_ids():
    """测试检索器返回文档ID"""
    print("\n" + "="*70)
    print("🧪 Test 3: Retriever with Document IDs")
    print("="*70)

    config_dict = {
        'retrieval_method': 'bge',
        'retrieval_model_path': 'BAAI/bge-large-en-v1.5',
        'corpus_path': '/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl',
        'index_path': '/home/user/FlashRAG/indexes/scifact',
        'retrieval_topk': 3,
        'retrieval_batch_size': 256,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'faiss_gpu': True,
        'use_sentence_transformer': False,
    }

    try:
        config = Config(config_dict=config_dict)
        retriever = get_retriever(config)

        # 测试查询
        test_query = "What is myelodysplasia?"
        print(f"\n📝 Test query: {test_query}")

        # 测试返回文档ID
        results, scores, doc_ids = retriever._batch_search(
            query=[test_query],
            num=3,
            return_score=True,
            return_doc_ids=True
        )

        print(f"\n📊 Results:")
        print(f"  - Retrieved {len(results[0])} documents")
        print(f"  - Document IDs: {doc_ids[0]}")
        print(f"  - Scores: {[f'{s:.4f}' for s in scores[0]]}")

        for idx, (doc, doc_id, score) in enumerate(zip(results[0], doc_ids[0], scores[0]), 1):
            print(f"\n  Document {idx} (ID: {doc_id}, Score: {score:.4f}):")
            title = doc.get('title', 'N/A')
            print(f"    Title: {title[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator_with_logits():
    """测试生成器返回logits"""
    print("\n" + "="*70)
    print("🧪 Test 4: Generator with Logits")
    print("="*70)

    config_dict = {
        'generator_model': 'llama3.1-8b-instruct',
        'generator_model_path': '/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct',
        'generator_max_input_len': 2048,
        'generator_batch_size': 1,
        'generation_params': {
            'do_sample': False,
        },
        'framework': 'hf',
        'device': 'cuda',
    }

    try:
        config = Config(config_dict=config_dict)
        generator = get_generator(config)

        # 测试输入
        test_input = "Answer the following question with only one letter (A, B, C, D, or E): What is the capital of France?"

        print(f"\n📝 Test input: {test_input}")

        # 生成并返回logits
        output = generator.generate(
            [test_input],
            return_dict=True,
            max_new_tokens=10
        )

        response = output['responses'][0]
        generated_logits = output['generated_token_logits'][0]  # [num_tokens, vocab_size]

        print(f"\n📊 Results:")
        print(f"  - Response: {response}")
        print(f"  - Logits shape: {generated_logits.shape}")
        print(f"  - First token logits shape: {generated_logits[0].shape}")

        # 获取 A-E 的 token IDs 和概率
        answer_tokens = ['A', 'B', 'C', 'D', 'E']
        answer_token_ids = {
            token: generator.tokenizer.convert_tokens_to_ids(token)
            for token in answer_tokens
        }

        print(f"\n  Answer token IDs: {answer_token_ids}")

        # 提取第一个token的概率
        first_token_logits = generated_logits[0]
        answer_probs = {
            token: first_token_logits[token_id].item()
            for token, token_id in answer_token_ids.items()
        }

        print(f"  Answer probabilities:")
        for token, prob in answer_probs.items():
            print(f"    {token}: {prob:.6f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_turn_chat():
    """测试多轮对话功能"""
    print("\n" + "="*70)
    print("🧪 Test 5: Multi-Turn Chat")
    print("="*70)

    try:
        from mia_multi_turn_chat import create_mia_chat

        # 创建对话实例
        chat = create_mia_chat()

        # 测试对话 1: 带检索
        print("\n📌 Turn 1: WITH retrieval")
        print("-" * 70)
        result1 = chat.chat(
            user_query="What is the role of MDSC in myelodysplasia?",
            use_retrieval=True,
            topk=3
        )

        # 测试对话 2: 不带检索
        print("\n📌 Turn 2: WITHOUT retrieval")
        print("-" * 70)
        result2 = chat.chat(
            user_query="Can you elaborate on that?",
            use_retrieval=False
        )

        # 打印对话历史
        chat.print_conversation_history()

        # 验证结果
        print("\n📊 Verification:")
        print(f"  - Turn 1 retrieved docs: {result1['retrieved_doc_ids']}")
        print(f"  - Turn 1 answer probs: {result1['answer_probs']}")
        print(f"  - Turn 2 response: {result2['response']}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_query_pipeline():
    """测试单个查询的完整pipeline"""
    print("\n" + "="*70)
    print("🧪 Test 6: Complete Single Query Pipeline")
    print("="*70)

    config_dict = {
        # Generator 配置
        'generator_model': 'llama3.1-8b-instruct',
        'generator_model_path': '/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct',
        'generator_max_input_len': 2048,
        'generator_batch_size': 1,
        'generation_params': {
            'do_sample': False,
        },

        # Retriever 配置
        'retrieval_method': 'bge',
        'retrieval_model_path': 'BAAI/bge-large-en-v1.5',
        'corpus_path': '/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl',
        'index_path': '/home/user/FlashRAG/indexes/scifact',
        'retrieval_topk': 3,
        'retrieval_batch_size': 256,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'faiss_gpu': True,
        'use_sentence_transformer': False,

        # 其他配置
        'framework': 'hf',
        'device': 'cuda',
    }

    try:
        config = Config(config_dict=config_dict)
        retriever = get_retriever(config)
        generator = get_generator(config)
        prompt_template = PromptTemplate(
            config,
            system_prompt=PromptTemplate.mia_system_prompt,
            user_prompt=PromptTemplate.base_user_prompt
        )

        # 测试查询
        test_query = "What is the function of BC1 RNA in ID element amplification?"

        print(f"\n📝 Query: {test_query}")

        # 1. 检索
        print("\n🔍 Step 1: Retrieval")
        results, scores, doc_ids = retriever._batch_search(
            query=[test_query],
            num=3,
            return_score=True,
            return_doc_ids=True
        )

        print(f"  Retrieved document IDs: {doc_ids[0]}")

        # 2. 构建 prompt
        print("\n📄 Step 2: Prompt Construction")
        input_prompt = prompt_template.get_string(
            question=test_query,
            retrieval_result=results[0]
        )
        print(f"  Prompt length: {len(input_prompt)} characters")

        # 3. 生成
        print("\n🤖 Step 3: Generation")
        output = generator.generate(
            [input_prompt],
            return_dict=True,
            max_new_tokens=10
        )

        response = output['responses'][0]
        logits = output['generated_token_logits'][0]

        print(f"  Response: {response}")
        print(f"  Logits shape: {logits.shape}")

        # 4. 分析答案
        print("\n📊 Step 4: Answer Analysis")
        answer_tokens = ['A', 'B', 'C', 'D', 'E']
        answer_token_ids = {
            token: generator.tokenizer.convert_tokens_to_ids(token)
            for token in answer_tokens
        }

        first_token_logits = logits[0]
        answer_probs = {
            token: first_token_logits[token_id].item()
            for token, token_id in answer_token_ids.items()
        }

        print("  Answer probabilities:")
        for token, prob in sorted(answer_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"    {token}: {prob:.6f}")

        print("\n✅ Complete pipeline test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 MIA Pipeline - Complete Testing Suite")
    print("="*70)

    tests = [
        # ("Data Preparation", test_data_preparation),
        # ("Index Building", test_index_building),
        ("Retriever with Doc IDs", test_retriever_with_doc_ids),
        ("Generator with Logits", test_generator_with_logits),
        ("Multi-Turn Chat", test_multi_turn_chat),
        ("Complete Single Query Pipeline", test_single_query_pipeline),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)}"

    # 打印总结
    print("\n" + "="*70)
    print("📋 Test Summary")
    print("="*70)
    for test_name, result in results.items():
        print(f"  {test_name}: {result}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
