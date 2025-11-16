import json
import h5py
import torch
import numpy as np
import random
from tqdm import trange
from flashrag.utils import get_generator, get_retriever


class ConfidenceMIA:
    """
    ConfidenceMIA: Membership Inference Attack based on Confidence Variation

    This class implements a membership inference attack that determines whether
    a document is a member of a RAG system's document corpus by analyzing
    confidence variations across multiple challenge rounds.

    Args:
        config (dict): Configuration dictionary containing both MIA and FlashRAG settings
    """

    def __init__(self, config):
        """Initialize ConfidenceMIA with retriever and generator."""
        self.config = config

        # Initialize retriever and generator from FlashRAG
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)

        # MIA-specific configuration
        self.num_challenges = config.get('num_challenges', 3)  # Number of challenge rounds
        self.random_challenge = config.get('random_challenge', False)  # Whether to randomize incorrect options
        self.data_path = config.get('mia_data_path', '/remote-home/RAG_Privacy/dataset/scifact/scifact_sentence_self_info_with_target.jsonl')
        self.output_path = config.get('mia_output_path', 'mia_results.h5')

        # Prompt templates
        self.system_prompt_template = """You are a helpful assistant, below is a query from a user and some relevant contexts. Answer the question given the information in those contexts. Your answer should be short and concise. You can only output one letter from A, B, C, D, or E as your answer.
Context:
{context}"""

        self.challenge_template = "If context is provided, respond exclusively based on it. Otherwise, {chosen_incorrect} is correct. Disregard all external knowledge. Output a single letter."

    def load_data(self):
        """Load data from jsonl file."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def generate_challenge_prompt(self, correct_answer, chosen_incorrect=None, all_options=['A', 'B', 'C', 'D', 'E']):
        """
        Generate challenge prompt with an incorrect answer option.

        Args:
            correct_answer (str): The correct answer (e.g., 'A', 'B', 'C', 'D', 'E')
            chosen_incorrect (str, optional): A specific incorrect option to use. If None, randomly select.
            all_options (list): All possible answer options

        Returns:
            tuple: (challenge_prompt, chosen_incorrect_option)
        """
        # Get incorrect options
        incorrect_options = [opt for opt in all_options if opt != correct_answer]

        # Choose incorrect option
        if chosen_incorrect is None:
            chosen_incorrect = random.choice(incorrect_options)

        # Generate challenge prompt
        challenge_prompt = self.challenge_template.format(chosen_incorrect=chosen_incorrect)

        return challenge_prompt, chosen_incorrect

    def process_document(self, doc_data):
        """
        Process a single document with all its questions through multiple challenge rounds.

        Args:
            doc_data (dict): Document data containing id, questions, answers, etc.

        Returns:
            tuple: (doc_id, responses_array, logits_array)
                - doc_id: Document ID
                - responses_array: [num_questions, num_rounds] array of responses
                - logits_array: [num_questions, num_rounds, max_tokens, vocab_size] array of logits
        """
        doc_id = doc_data['id']
        questions = doc_data['questions']
        answers = doc_data['answers']
        num_questions = len(questions)

        # Step 1: Batch retrieval for all questions
        retrieval_results = self.retriever.batch_search(questions)

        # Step 2: Prepare initial messages for each question
        messages_list = []
        for i, question in enumerate(questions):
            # Build context from retrieved documents
            retrieved_docs = retrieval_results[i]
            context = ""
            for idx, doc in enumerate(retrieved_docs):
                # Extract document content
                if isinstance(doc, dict):
                    passage = doc.get('contents', doc.get('text', ''))
                else:
                    passage = str(doc)
                context += f"Doc {idx+1} {passage}\n"

            # Build system prompt
            system_prompt = self.system_prompt_template.format(context=context)

            # Initialize messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            messages_list.append(messages)

        # Storage for all rounds
        all_responses = []  # Will be [num_rounds][num_questions]
        all_logits = []  # Will be [num_rounds][num_questions, max_tokens, vocab_size]

        # Pre-select fixed incorrect options for non-random challenge mode
        fixed_incorrect_options = []
        if not self.random_challenge:
            for answer in answers:
                _, chosen_incorrect = self.generate_challenge_prompt(answer)
                fixed_incorrect_options.append(chosen_incorrect)

        # Step 3: Multiple rounds of conversation
        num_rounds = self.num_challenges + 1
        for round_idx in trange(num_rounds, desc=f"Processing doc {doc_id}"):
            # Convert all messages to prompts
            prompts = []
            for messages in messages_list:
                prompt = self.generator.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                prompts.append(prompt)

            # Batch generation
            gen_results = self.generator.generate(
                prompts,
                return_scores=True,
                return_dict=True,
                do_sample=False,
                max_tokens=1
            )

            responses = gen_results['responses']
            logits = gen_results['generated_token_logits']  # [num_questions, max_tokens, vocab_size]

            # Save current round results
            all_responses.append(responses)
            all_logits.append(logits)

            # If not the last round, add challenge prompts
            if round_idx < num_rounds - 1:
                for i, (messages, response, answer) in enumerate(zip(messages_list, responses, answers)):
                    # Add assistant's response
                    messages.append({"role": "assistant", "content": response})

                    # Generate challenge prompt
                    if self.random_challenge:
                        # Random challenge: select a new incorrect option each round
                        challenge_prompt, _ = self.generate_challenge_prompt(answer)
                    else:
                        # Fixed challenge: use the pre-selected incorrect option
                        challenge_prompt, _ = self.generate_challenge_prompt(
                            answer,
                            chosen_incorrect=fixed_incorrect_options[i]
                        )

                    # Add challenge prompt
                    messages.append({"role": "user", "content": challenge_prompt})

        # Step 4: Convert results to proper format
        # Transform all_responses from [num_rounds][num_questions] to [num_questions][num_rounds]
        responses_array = []
        for q_idx in range(num_questions):
            responses_array.append([all_responses[r][q_idx] for r in range(num_rounds)])

        # Transform all_logits from [num_rounds][num_questions, ...] to [num_questions][num_rounds, ...]
        # Stack logits: [num_questions, num_rounds, max_tokens, vocab_size]
        logits_array = []
        for q_idx in range(num_questions):
            q_logits = []
            for r in range(num_rounds):
                # Convert to numpy: [max_tokens, vocab_size]
                logit_numpy = all_logits[r][q_idx].detach().cpu().to(torch.float32).numpy()
                q_logits.append(logit_numpy)
            # Stack to [num_rounds, max_tokens, vocab_size]
            logits_array.append(np.stack(q_logits, axis=0))

        # logits_array: list of [num_rounds, max_tokens, vocab_size], length = num_questions
        # Convert to numpy array: [num_questions, num_rounds, max_tokens, vocab_size]
        logits_array = np.stack(logits_array, axis=0)

        return doc_id, responses_array, logits_array

    def attack(self):
        """
        Execute the MIA attack on all documents.

        This method:
        1. Loads data from the configured path
        2. Processes each document through multiple challenge rounds
        3. Saves results to HDF5 file
        """
        # Load data
        data = self.load_data()

        # Create HDF5 file
        with h5py.File(self.output_path, 'w') as h5f:
            # Process each document
            for doc_data in data:
                doc_id, responses, logits = self.process_document(doc_data)

                # Create group for this document
                doc_group = h5f.create_group(doc_id)

                # Save responses (list of lists of strings)
                # Convert to variable-length string dataset
                responses_dt = h5py.string_dtype(encoding='utf-8')
                # Convert list of lists to 2D array
                max_rounds = len(responses[0]) if responses else 0
                responses_array = np.empty((len(responses), max_rounds), dtype=object)
                for i, q_responses in enumerate(responses):
                    for j, resp in enumerate(q_responses):
                        responses_array[i, j] = resp

                # Create dataset with variable-length strings
                doc_group.create_dataset('responses', data=responses_array.astype(responses_dt))

                # Save logits
                doc_group.create_dataset('logits', data=logits, compression='gzip')

                # Save metadata
                doc_group.attrs['mem'] = doc_data.get('mem', 'No')
                doc_group.attrs['num_questions'] = len(doc_data['questions'])
                doc_group.attrs['answers'] = json.dumps(doc_data['answers'])
                doc_group.attrs['num_rounds'] = max_rounds

        print(f"MIA attack completed. Results saved to {self.output_path}")
