"""
Example usage of ConfidenceMIA for membership inference attack.

This script demonstrates how to use the ConfidenceMIA class to perform
membership inference attacks on RAG systems.
"""

from flashrag.mia import ConfidenceMIA

# Configuration for ConfidenceMIA and FlashRAG components
config = {
    # ========== Retriever Settings ==========
    "use_multi_retriever": False,
    "retrieval_method": "bge",  # Name or path of the retrieval model
    "retrieval_model_path": "/remote-home/RAG_Privacy/model/BAAI/bge-large-en-v1.5",
    "index_path": "/remote-home/RAG_Privacy/index/bge_Flat.index",
    "faiss_gpu": True,  # Whether to use GPU for FAISS index
    "corpus_path": "/remote-home/RAG_Privacy/dataset/scifact/corpus.jsonl",
    "instruction": "Represent this sentence for searching relevant passages: ",
    "retrieval_topk": 5,  # Number of retrieved documents
    "retrieval_batch_size": 256,
    "retrieval_use_fp16": True,
    "retrieval_query_max_length": 128,
    "save_retrieval_cache": False,
    "use_retrieval_cache": False,
    "retrieval_cache_path": None,
    "retrieval_pooling_method": 'cls',
    "use_sentence_transformer": False,
    "use_reranker": False,

    # ========== Dataset Settings ==========
    "data_dir": "/remote-home/RAG_Privacy/dataset/",
    "dataset_name": "scifact",
    "dataset_path": "/remote-home/RAG_Privacy/dataset/scifact",
    "split": ["train"],
    "test_sample_num": None,
    "random_sample": False,

    # ========== Generator Settings ==========
    "framework": "hf",
    "generator_model": "llama3-8B-instruct",
    "generator_model_path": "/remote-home/RAG_Privacy/model/meta-llama/Llama-3.1-8B-Instruct",
    "generator_max_input_len": 1024,
    "generator_batch_size": 4,
    "generation_params": {
        "do_sample": False,
        "max_tokens": 1
    },
    "use_fid": False,
    "device": "cuda:0",
    "gpu_num": 1,
    "seed": 42,

    # ========== MIA-Specific Settings ==========
    "num_challenges": 3,  # Number of challenge rounds (total rounds = num_challenges + 1)
    "random_challenge": False,  # If False, use fixed incorrect option; if True, randomize each round
    "mia_data_path": "/remote-home/RAG_Privacy/dataset/scifact/scifact_sentence_self_info_with_target.jsonl",
    "mia_output_path": "mia_results.h5"  # Path to save HDF5 results
}


def main():
    """Run the ConfidenceMIA attack."""

    print("=" * 80)
    print("ConfidenceMIA - Membership Inference Attack for RAG Systems")
    print("=" * 80)
    print()

    print(f"Configuration:")
    print(f"  - Retrieval Model: {config['retrieval_method']}")
    print(f"  - Generator Model: {config['generator_model']}")
    print(f"  - Number of Challenge Rounds: {config['num_challenges']}")
    print(f"  - Random Challenge Mode: {config['random_challenge']}")
    print(f"  - Data Path: {config['mia_data_path']}")
    print(f"  - Output Path: {config['mia_output_path']}")
    print()

    # Initialize ConfidenceMIA
    print("Initializing ConfidenceMIA...")
    mia = ConfidenceMIA(config)
    print("Initialization complete!")
    print()

    # Run the attack
    print("Starting MIA attack...")
    mia.attack()
    print()

    print("=" * 80)
    print("Attack completed successfully!")
    print(f"Results saved to: {config['mia_output_path']}")
    print("=" * 80)


def read_results(h5_path="mia_results.h5"):
    """
    Example function to read and analyze the HDF5 results.

    Args:
        h5_path (str): Path to the HDF5 results file
    """
    import h5py
    import json

    print("=" * 80)
    print("Reading MIA Results")
    print("=" * 80)
    print()

    with h5py.File(h5_path, 'r') as h5f:
        print(f"Number of documents processed: {len(h5f.keys())}")
        print()

        # Analyze first document as example
        doc_id = list(h5f.keys())[0]
        doc_group = h5f[doc_id]

        print(f"Example Document: {doc_id}")
        print(f"  - Is Member: {doc_group.attrs['mem']}")
        print(f"  - Number of Questions: {doc_group.attrs['num_questions']}")
        print(f"  - Number of Rounds: {doc_group.attrs['num_rounds']}")
        print(f"  - Correct Answers: {json.loads(doc_group.attrs['answers'])}")
        print()

        # Load responses and logits
        responses = doc_group['responses'][:]
        logits = doc_group['logits'][:]

        print(f"Responses shape: {responses.shape}")  # [num_questions, num_rounds]
        print(f"Logits shape: {logits.shape}")  # [num_questions, num_rounds, max_tokens, vocab_size]
        print()

        # Show responses for first question
        print("First question responses across all rounds:")
        for round_idx, response in enumerate(responses[0]):
            response_text = response.decode('utf-8') if isinstance(response, bytes) else response
            print(f"  Round {round_idx + 1}: {response_text}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    # Run the attack
    main()

    # Optionally, read and analyze results
    # read_results(config['mia_output_path'])
