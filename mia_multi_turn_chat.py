"""
多轮对话接口 - 用于成员推理攻击实验
支持：
1. 选择是否检索
2. 返回文档 ID
3. 返回 logits
4. 维护对话历史
"""

import torch
from typing import List, Dict, Optional, Tuple
from flashrag.config import Config
from flashrag.utils import get_generator, get_retriever
from flashrag.prompt import PromptTemplate
from flashrag.retriever.utils import load_corpus


class MIAMultiTurnChat:
    """多轮对话类，支持成员推理攻击实验"""

    def __init__(self, config):
        """
        初始化多轮对话系统

        Args:
            config: FlashRAG 配置对象
        """
        self.config = config
        self.generator = get_generator(config)
        self.retriever = get_retriever(config)

        # 使用 MIA 专用的提示词
        self.prompt_template = PromptTemplate(
            config,
            system_prompt=PromptTemplate.mia_system_prompt,
            user_prompt=PromptTemplate.base_user_prompt
        )

        # 不带检索的提示词模板
        self.no_retrieval_prompt_template = PromptTemplate(
            config,
            system_prompt="You are a helpful assistant. Answer the question based on your knowledge. "
                         "Your answer should be short and concise. You can only output one letter from A, B, C, D, or E as your answer.",
            user_prompt="Question: {question}"
        )

        # 初始化对话历史
        self.messages = []

        # 获取答案 token IDs (A, B, C, D, E)
        self.answer_tokens = ['A', 'B', 'C', 'D', 'E']
        self.answer_token_ids = {
            token: self.generator.tokenizer.convert_tokens_to_ids(token)
            for token in self.answer_tokens
        }
        # 反向映射
        self.inv_answer_token_ids = {v: k for k, v in self.answer_token_ids.items()}

        print("✅ MIA Multi-Turn Chat initialized successfully!")

    def reset_conversation(self):
        """重置对话历史"""
        self.messages = []
        print("🔄 Conversation history reset")

    def format_documents(self, documents: List[Dict]) -> str:
        """
        格式化检索到的文档

        Args:
            documents: 文档列表

        Returns:
            格式化的文档字符串
        """
        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            # 处理不同的文档格式
            if 'text' in doc:
                content = doc['text']
            elif 'contents' in doc:
                content = doc['contents']
            else:
                content = str(doc)

            formatted_docs.append(f"Document {idx}:\n{content}\n")

        return "\n".join(formatted_docs)

    def chat(
        self,
        user_query: str,
        use_retrieval: bool = True,
        topk: int = 3,
        return_details: bool = True
    ) -> Dict:
        """
        执行一轮对话

        Args:
            user_query: 用户查询
            use_retrieval: 是否使用检索
            topk: 检索文档数量
            return_details: 是否返回详细信息（文档ID、logits等）

        Returns:
            包含回答和详细信息的字典
        """
        result = {
            'user_query': user_query,
            'use_retrieval': use_retrieval,
            'response': '',
            'retrieved_doc_ids': None,
            'retrieved_documents': None,
            'logits': None,
            'answer_probs': None,
            'predicted_answer': None
        }

        # 1. 检索（如果需要）
        if use_retrieval:
            print(f"🔍 Retrieving top-{topk} documents...")

            # 使用修改后的检索器，返回文档ID
            retrieved_docs, scores, doc_ids = self.retriever._batch_search(
                query=[user_query],
                num=topk,
                return_score=True,
                return_doc_ids=True
            )

            retrieved_docs = retrieved_docs[0]  # 获取第一个查询的结果
            doc_ids = doc_ids[0]
            scores = scores[0]

            result['retrieved_doc_ids'] = doc_ids
            result['retrieved_documents'] = retrieved_docs

            print(f"📄 Retrieved document IDs: {doc_ids}")

            # 格式化文档
            formatted_docs = self.format_documents(retrieved_docs)

            # 构建包含检索结果的 prompt
            input_prompt = self.prompt_template.get_string(
                question=user_query,
                retrieval_result=retrieved_docs
            )

            # 添加到消息历史
            self.messages.append({
                "role": "user",
                "content": user_query,
                "retrieved_docs": doc_ids  # 保存检索到的文档ID
            })

        else:
            print("💬 Direct generation without retrieval...")

            # 不使用检索，直接添加用户查询
            self.messages.append({
                "role": "user",
                "content": user_query
            })

            # 使用不带检索的提示词
            input_prompt = self.no_retrieval_prompt_template.get_string(
                question=user_query
            )

        # 2. 生成回答
        print("🤖 Generating response...")

        # 使用 return_dict=True 获取 logits
        generation_output = self.generator.generate(
            [input_prompt],
            return_dict=True,
            max_new_tokens=10  # 因为只需要输出一个字母
        )

        response = generation_output['responses'][0]
        generated_token_logits = generation_output['generated_token_logits'][0]  # [num_tokens, vocab_size]

        result['response'] = response

        # 添加助手回复到历史
        self.messages.append({
            "role": "assistant",
            "content": response
        })

        # 3. 提取答案 logits 和概率
        if return_details:
            # 获取第一个生成的 token 的 logits（即答案）
            first_token_logits = generated_token_logits[0]  # [vocab_size]

            # 提取 A-E 的概率
            answer_probs = {}
            for token, token_id in self.answer_token_ids.items():
                answer_probs[token] = first_token_logits[token_id].item()

            result['logits'] = first_token_logits.cpu()
            result['answer_probs'] = answer_probs

            # 预测的答案
            first_token_id = self.generator.tokenizer.encode(
                response,
                add_special_tokens=False
            )[0] if response else None

            predicted_answer = self.inv_answer_token_ids.get(first_token_id, response[0] if response else "N/A")
            result['predicted_answer'] = predicted_answer

            print(f"📊 Answer probabilities: {answer_probs}")
            print(f"✨ Predicted answer: {predicted_answer}")

        print(f"💡 Response: {response}\n")

        return result

    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.messages

    def print_conversation_history(self):
        """打印对话历史"""
        print("\n" + "="*50)
        print("📜 Conversation History")
        print("="*50)
        for idx, msg in enumerate(self.messages, 1):
            role = msg['role'].upper()
            content = msg['content']
            print(f"\n[{idx}] {role}:")
            print(f"  {content}")
            if 'retrieved_docs' in msg:
                print(f"  📎 Retrieved Docs: {msg['retrieved_docs']}")
        print("="*50 + "\n")


def create_mia_chat(
    model_path: str = "/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct",
    retrieval_method: str = "bge",
    corpus_path: str = "/home/user/FlashRAG/datasets/scifact/index_corpus.jsonl",
    index_path: str = "/home/user/FlashRAG/indexes/scifact",
    retrieval_topk: int = 3
) -> MIAMultiTurnChat:
    """
    创建 MIA 多轮对话实例的便捷函数

    Args:
        model_path: LLM 模型路径
        retrieval_method: 检索方法
        corpus_path: 语料库路径
        index_path: 索引路径
        retrieval_topk: 检索文档数量

    Returns:
        MIAMultiTurnChat 实例
    """
    config_dict = {
        # Generator 配置
        'generator_model': 'llama3.1-8b-instruct',
        'generator_model_path': model_path,
        'generator_max_input_len': 2048,
        'generator_batch_size': 1,
        'generation_params': {
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': False,  # 确定性生成
        },

        # Retriever 配置
        'retrieval_method': retrieval_method,
        'retrieval_model_path': 'BAAI/bge-large-en-v1.5',
        'corpus_path': corpus_path,
        'index_path': index_path,
        'retrieval_topk': retrieval_topk,
        'retrieval_batch_size': 256,
        'retrieval_pooling_method': 'mean',
        'retrieval_use_fp16': True,
        'faiss_gpu': True,
        'use_sentence_transformer': False,

        # 其他配置
        'framework': 'hf',
        'device': 'cuda',
        'gpu_id': 0,
    }

    config = Config(config_dict=config_dict)
    return MIAMultiTurnChat(config)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 创建对话实例
    chat = create_mia_chat()

    print("\n" + "="*70)
    print("🎯 MIA Multi-Turn Chat - Membership Inference Attack Experiments")
    print("="*70 + "\n")

    # 示例 1: 带检索的对话
    print("📌 Example 1: Chat WITH retrieval")
    print("-" * 70)
    result1 = chat.chat(
        user_query="What is the role of myeloid-derived suppressor cells in myelodysplasia?",
        use_retrieval=True,
        topk=3
    )

    # 示例 2: 不带检索的对话
    print("\n📌 Example 2: Chat WITHOUT retrieval")
    print("-" * 70)
    result2 = chat.chat(
        user_query="Can you explain more about that?",
        use_retrieval=False
    )

    # 示例 3: 再次带检索的对话
    print("\n📌 Example 3: Another query WITH retrieval")
    print("-" * 70)
    result3 = chat.chat(
        user_query="How does diffusion tensor MRI assess cerebral white matter?",
        use_retrieval=True,
        topk=3
    )

    # 打印完整的对话历史
    chat.print_conversation_history()

    # 重置对话
    chat.reset_conversation()

    print("\n✅ All examples completed!")
