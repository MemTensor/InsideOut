import os
import json
import argparse
import re
import torch
from tqdm import tqdm
import csv
from PersonaMem.prepare_blocks import *
from src.llm import Agent
from src.segment import split_conversation
import numpy as np
from typing import Any
from typing import Dict, List, Tuple, Union
from FlagEmbedding import FlagModel
from FlagEmbedding import BGEM3FlagModel
from FlagEmbedding import FlagReranker
from tqdm import tqdm
import json
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import time


MODEL_PATHS = {
    'bge_large': 'BAAI/bge-large-en-v1.5',
    'bge_m3': 'BAAI/bge-m3',
    'bge_reranker_large': 'BAAI/bge-reranker-large',
    'Qwen3-Reranker-4B': 'Qwen3-Reranker-4B',
    'Qwen3-Embedding-4B': "BAAI/Qwen3-Embedding-4B"
}
models = {}
# This function will be called once by each process in the pool to load the models.
def init_worker(rerank=False,rerank_model='bge_reranker_large'):
    """Initializes models in a worker process."""
    global models
    if not models: # Load models only if they haven't been loaded in this process yet
        print("Initializing models in a new worker process...",flush=True)
        models['bge_large'] = FlagModel(MODEL_PATHS['bge_large'], use_fp16=False)
        models['bge_m3'] = BGEM3FlagModel(MODEL_PATHS['bge_m3'], use_fp16=False)
        if rerank:
            if rerank_model=='bge_reranker_large':
                models['bge_reranker_large'] = FlagReranker(MODEL_PATHS['bge_reranker_large'], use_fp16=False)
            elif rerank_model=='Qwen3-Reranker-4B':
                models['Qwen3-Reranker-4B_tokenizer'] = AutoTokenizer.from_pretrained(MODEL_PATHS['Qwen3-Reranker-4B'], padding_side='left')
                models['Qwen3-Reranker-4B_model'] = AutoModelForCausalLM.from_pretrained(MODEL_PATHS['Qwen3-Reranker-4B'],device_map="auto", torch_dtype=torch.float32).eval()
            elif rerank_model=='Qwen3-Embedding-4B':
                models['Qwen3-Embedding-4B'] = SentenceTransformer(MODEL_PATHS['Qwen3-Embedding-4B'])
        print("Models initialized.",flush=True)



class Evaluation:
    def __init__(self):
        self.agent = Agent("You are a helpful assistant")
        


    def query_llm(self, question, all_options, context=None, instructions=None):
        """
        使用封装好的 Agent 调用模型。
        - 输入参数接口保持不变（question, all_options, context, instructions）。
        - 返回值仍然是模型的文本响应（str），以兼容后续 extract_answer / run_evaluation。
        """
        if instructions is None:
            instructions = (
                "你是一个根据用户个性化对话历史和用户属性树，从候选选项中选择最优回答的决策模块。\n\n"
                "你将看到：\n"
                "1. 用户问题；\n"
                "2. 若干候选回答选项；\n"
                "3. 该用户的对话历史；\n"
                "4. 用户属性树。\n\n"
                "你的目标是：**充分利用用户对话历史和用户属性树中的信息，为这个用户的问题选出最合适的一条选项回答**。\n\n"
                "请严格遵循以下规则进行选择：\n"
                "- 明确使用对话历史和用户属性树中的信息来对比各个选项的适配度；\n"
                "- 如果多个选项在通用场景下都合理，优先选择**最符合该用户对话历史、属性树和个性化需求**的选项；\n"
                "- 不要编造新的选项，不要修改候选选项的内容；\n"
                "- 不要输出任何分析过程或解释，只输出最终答案；\n"
                "- 最终答案格式必须为：<final_answer>(x)，其中 x ∈ {a, b, c, d}。"
            )
            
        prompt = (
            f"{instructions}\n\n"
            f"用户问题：\n{question}\n\n"
            f"候选回答选项:\n{all_options}\n\n"
            f"用户对话历史:\n{context}"
        )

        response, success = self.agent.run(prompt)
        if success:
            # print("model prompt:", prompt)
            print("model success:", success)
            print("model response:", response)
            return response
        else:
            return '<final_answer>(c)'



    def extract_answer(self, predicted_answer, correct_answer):
        def _extract_only_options(text):
            text = text.lower()
            in_parens = re.findall(r'\(([a-d])\)', text)
            if in_parens:
                return set(in_parens)
            else:
                return set(re.findall(r'\b([a-d])\b', text))

        correct = correct_answer.lower().strip("() ")

        # Clean predicted_answer
        full_response = predicted_answer
        predicted_answer = predicted_answer.strip()
        if "<final_answer>" in predicted_answer:
            predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
        if predicted_answer.endswith("</final_answer>"):
            predicted_answer = predicted_answer[:-len("</final_answer>")].strip()

        pred_options = _extract_only_options(predicted_answer)

        # First try the predicted_answer
        if pred_options == {correct}:
            return True, predicted_answer

        # Optionally fallback to model_response if provided
        response_options = _extract_only_options(full_response)
        if response_options == {correct}:
            return True, predicted_answer

        return False, predicted_answer



def load_rows_with_context(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value
            yield row_data


def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header row



def build_embedding_index(
    corpus: Dict,
    model_name: str = "bge_large"
) -> Dict[Union[str, int], Dict[str, Any]]:
    """
    为字典中所有文本块预先计算 embedding，返回索引。

    返回结构:
    {
        标号1: {
            "texts": [...],
            "embeddings": np.ndarray (num_texts, dim)
        },
        标号2: {
            ...
        }
    }
    """
    index = {}
    model = models[model_name]

    for label, texts in corpus.items():
        if not texts:
            continue

        all_embs: List[np.ndarray] = []

        batch_size=16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            if model_name == "bge_large":
                batch_embs = model.encode(batch_texts) 
            elif model_name == "bge_m3":
                batch_embs = model.encode(batch_texts)['dense_vecs']
            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

            batch_embs = np.array(batch_embs, dtype="float32")
            all_embs.append(batch_embs)

        embs = np.vstack(all_embs)
        
        index[label] = {
            "texts": texts,
            "embeddings": embs
        }
    return index

def search_with_index(
    index: Dict[Union[str, int], Dict[str, Any]],
    label: Union[str, int],
    query: str,
    n: int,
    model_name: str = "bge_large"
) -> List[Tuple[str, float]]:
    """
    在预先构建的向量索引上做检索：
    给定标号 label + query + n，返回最相关的 n 个文本块。
    """
    if label not in index:
        return []

    entry = index[label]
    texts = entry["texts"]
    embs = entry["embeddings"]  

    if embs.size == 0:
        return []

    model = models[model_name]
    if model_name == "bge_large":
        q_emb = model.encode([query]) 
    elif model_name == "bge_m3":
        q_emb = model.encode([query])['dense_vecs'] 
    q_emb = np.array(q_emb, dtype="float32")[0] 

    scores = embs @ q_emb 
    n = min(n, len(texts))
    top_indices = np.argsort(scores)[::-1][:n]

    result = [(texts[i], float(scores[i])) for i in top_indices]
    return result

def keep_leaf_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归清洗字典：
    - 保留叶子节点为非空字符串的项；
    - 删除空值；
    - 保留有效叶子的路径。
    """
    if isinstance(d, str):
        return d if d.strip() != "" else None

    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            cleaned = keep_leaf_keys(value)
            if cleaned is not None:  
                new_dict[key] = cleaned

        return new_dict if new_dict else None
    return None

def retrieve_and_rerank(
    dialogue_index,
    shared_context_id,
    questions,         
    top_n,
    model_name,
    rerank_model_name,
    query_text,    
):
    """
    返回针对同一个问题，经多query检索+向量重排后的前 top_n 个 chunk。
    """

    all_chunks = []
    questions=[query_text]+questions
    for rewritten_question in questions:
        top_chunks = search_with_index(dialogue_index,shared_context_id,rewritten_question,min(top_n,4),model_name)
        all_chunks.extend(top_chunks)
    all_chunks=[chunk for chunk, _ in all_chunks]
    if not all_chunks:
        return []

    unique_chunks = list(dict.fromkeys(all_chunks))

    if rerank_model_name == 'bge_reranker_large':
        rerank_model = models['bge_reranker_large']
        batch_size=16
        pairs = [[query_text, chunk] for chunk in unique_chunks]
        scores = rerank_model.compute_score(pairs, batch_size=batch_size)

        if isinstance(scores, list):
            scores = np.array(scores)

        n = min(top_n, len(unique_chunks))
        top_indices = np.argsort(scores)[::-1][:n]
        reranked_chunks = [(unique_chunks[i], float(scores[i])) for i in top_indices]
    elif rerank_model_name == 'Qwen3-Reranker-4B':
        rerank_model = models['Qwen3-Reranker-4B_model']
        rerank_tokenizer = models['Qwen3-Reranker-4B_tokenizer']
        token_false_id = rerank_tokenizer.convert_tokens_to_ids("no")
        token_true_id = rerank_tokenizer.convert_tokens_to_ids("yes")
        max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = rerank_tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = rerank_tokenizer.encode(suffix, add_special_tokens=False)
        def format_instruction(instruction, query, doc):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
            return output

        def process_inputs(pairs):
            inputs = rerank_tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            inputs = rerank_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
            for key in inputs:
                inputs[key] = inputs[key].to(rerank_model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs, **kwargs):
            batch_scores = rerank_model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        qa_pairs = [[query_text, chunk] for chunk in unique_chunks]

        scores = []
        for query, doc in qa_pairs: 
            # Tokenize the input texts
            inputs = process_inputs([format_instruction(task, query, doc)])
            score = compute_logits(inputs)[0]
            scores.append(score)

        if isinstance(scores, list):
            scores = np.array(scores)

        n = min(top_n, len(unique_chunks))
        top_indices = np.argsort(scores)[::-1][:n]
        reranked_chunks = [(unique_chunks[i], float(scores[i])) for i in top_indices]
    elif rerank_model_name == 'Qwen3-Embedding-4B':
        rerank_model=models['Qwen3-Embedding-4B']
        query_embeddings = rerank_model.encode([query_text], prompt_name="query")
        q_emb = query_embeddings[0]
        all_embs: List[np.ndarray] = []

        batch_size=16
        for i in range(0, len(unique_chunks), batch_size):
            batch_texts = unique_chunks[i:i + batch_size]
            batch_embs = rerank_model.encode(batch_texts)  
            batch_embs = np.array(batch_embs, dtype="float32")
            all_embs.append(batch_embs)
        embs = np.vstack(all_embs)
        scores = embs @ q_emb  
        n = min(top_n, len(unique_chunks))
        top_indices = np.argsort(scores)[::-1][:n]
        reranked_chunks = [(unique_chunks[i], float(scores[i])) for i in top_indices]
    return reranked_chunks


def run_evaluation(cmd_args, llm):
    question_path = cmd_args.question_path
    dialogue_path = cmd_args.dialogue_path
    model_name = cmd_args.model_name
    top_n = cmd_args.top_n
    if_rerank = cmd_args.rerank
    rerank_model_name = cmd_args.rerank_model
    memtree_path = cmd_args.memtree_path
    result_path = cmd_args.result_path
    
    rewrite_agent = Agent("You are a helpful assistant.")

    if os.path.exists(result_path):
        os.remove(result_path)

    all_errors = []
    total_rows = count_csv_rows(question_path)
    dialogue_all = {}
    
    with open(memtree_path, 'r', encoding='utf-8') as f:
        memtree_all = json.load(f)

    with open(dialogue_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            obj = json.loads(line)  
            dialogue_all.update(obj)
    dialogue_chunks={}
    for key, value in dialogue_all.items():    
        final_chunks=split_conversation(value,1)
        final_chunks=[json.dumps(chunk, ensure_ascii=False,indent=4) for chunk in final_chunks]
        dialogue_chunks[key]=final_chunks
    dialogue_index = build_embedding_index(dialogue_chunks, model_name)

    for row_data in tqdm(load_rows_with_context(question_path), total=total_rows):
        time.sleep(2)
        try:
            # Extract relevant data from the row
            persona_id = row_data["persona_id"]
            question_id = row_data["question_id"]
            question_type = row_data["question_type"]
            topic = row_data["topic"]
            context_length_in_tokens = row_data["context_length_in_tokens"]
            context_length_in_letters = row_data["context_length_in_letters"]
            distance_to_ref_in_blocks = row_data["distance_to_ref_in_blocks"]
            distance_to_ref_in_tokens = row_data["distance_to_ref_in_tokens"]
            num_irrelevant_tokens = row_data["num_irrelevant_tokens"]
            distance_to_ref_proportion_in_context = row_data["distance_to_ref_proportion_in_context"]
            question = row_data["user_question_or_message"]
            correct_answer = row_data["correct_answer"]
            all_options = row_data["all_options"]
            shared_context_id = row_data["shared_context_id"]
            end_index_in_shared_context = row_data["end_index_in_shared_context"]
            
            
            # Prepare the context for the LLM query
            processed = keep_leaf_keys(memtree_all.get(shared_context_id, None))
            context_memtree = json.dumps(processed, indent=2, ensure_ascii=False).strip()
            context="None\n\n用户属性树：\n"+context_memtree
            
            # Send the query to the LLM
            model_response = llm.query_llm(question, all_options, context)
            score, predicted_answer = llm.extract_answer(model_response, correct_answer)
            
            if correct_answer in predicted_answer:
                pass
            else:
                if if_rerank:
                    rewrite_prompt = f"""你是一个检索问题改写助手。

    目标：根据「用户属性树」和「原始问题」，生成若干个用于召回候选内容的英文检索查询句子。

    要求：
    1. 输出内容使用专业的英文；
    2. 根据用户属性树和原始问题，推测可能需要的信息，生成 2~3 个检索查询；
    3. 仅在用户属性树中出现的个性化内容与原始问题明显相关时，才将这些词汇适度融入查询。不要根据用户属性树臆造原问题中没有的约束；
    4. 必须保留原始问题中的核心名词和关键短语。可以适度加入含义接近的同义词或常用表达，用于问题扩展；
    5. 不同查询之间要有一定差异，避免完全重复，但都必须围绕原始问题；
    6. 输出时只返回若干个检索查询句子，每个占一行，不要输出任何分析过程或解释。


    用户属性树：
    {context_memtree}


    原始问题：
    {question}
    """

                    try:
                        response, success = rewrite_agent.run(rewrite_prompt)
                        if success:
                            print('Rewrite Questions:',response)
                            rewritten_questions = response.strip()
                            questions=[i.strip() for i in rewritten_questions.split('\n') if i.strip()!='']
                            top_chunks = retrieve_and_rerank(dialogue_index, shared_context_id, questions, top_n, model_name, rerank_model_name,question) 
                        else:
                            top_chunks = search_with_index(dialogue_index, shared_context_id, question, top_n, model_name)
                    except Exception as e:
                        top_chunks = search_with_index(dialogue_index, shared_context_id, question, top_n, model_name)
                        print("Rewrite Error:", e)
                else:
                    top_chunks = search_with_index(dialogue_index, shared_context_id, question, top_n, model_name)
                context=''
                for chunk, score in top_chunks:
                    context += chunk + "\n\n"
                context+="\n\n用户属性树：\n"+context_memtree
                
                # Send the query to the LLM
                model_response = llm.query_llm(question, all_options, context)
                score, predicted_answer = llm.extract_answer(model_response, correct_answer)
            
            # Save the results back to a CSV file together with the question types
            print(f"Question: {question}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Score: {score}")
            
            

            with open(result_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write the header if the file is empty
                if os.stat(result_path).st_size == 0:
                    writer.writerow(["score", "persona_id", "question_id", "user_question_or_message", "question_type", "topic", "context_length_in_tokens", "context_length_in_letters",
                                     "distance_to_ref_in_blocks", "distance_to_ref_in_tokens", "num_irrelevant_tokens", "distance_to_ref_proportion_in_context",
                                     "model_response", "len_of_model_response", "predicted_answer", "correct_answer"])
                writer.writerow([
                    score,
                    persona_id,
                    question_id,
                    question,
                    question_type,
                    topic,
                    context_length_in_tokens,
                    context_length_in_letters,
                    distance_to_ref_in_blocks,
                    distance_to_ref_in_tokens,
                    num_irrelevant_tokens,
                    distance_to_ref_proportion_in_context,
                    model_response,
                    len(model_response),
                    predicted_answer,
                    correct_answer,
                ])
        except Exception as e:
            print(f"Error: {e}")
            all_errors.append({
                "persona_id": row_data["persona_id"],
                "question_id": row_data["question_id"],
                "error": str(e)
            })
            continue

    if all_errors:
        for error in all_errors:
            print(f"Error for persona_id {error['persona_id']} and question_id {error['question_id']}: {error['error']}")


if __name__ == "__main__":
    torch.manual_seed(0)
    world_size = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--question_path', type=str, default='data/questions_32k.csv', help='Path to the questions CSV file')
    parser.add_argument('--dialogue_path', type=str, default='data/shared_contexts_32k.jsonl', help='Path to the dialogues JSONL file')
    parser.add_argument('--model_name', type=str, default='bge_m3', help='Model name to use for embeddings')
    parser.add_argument('--top_n', type=int, default=4, help='Number of top chunks to retrieve')     
    parser.add_argument('--rerank', type=bool, default=True, help='Whether to rerank the top chunks')  
    parser.add_argument('--rerank_model', type=str, default='bge_reranker_large', help='Model name to use for reranking')  
    parser.add_argument('--memtree_path', type=str, default='', help='Path to the memory tree JSON file')
    parser.add_argument('--result_path', type=str, default='results/Memrewriter/eval_results_32k.csv', help='Path to save the results CSV file')

    cmd_args = parser.parse_args()

    init_worker(rerank=cmd_args.rerank,rerank_model=cmd_args.rerank_model)
    llm = Evaluation()

    run_evaluation(cmd_args, llm)



# CUDA_VISIBLE_DEVICES=3 nohup python -u infer_memrewriter.py  > results/Memrewriter/eval_results_32k.log 2>&1 &

