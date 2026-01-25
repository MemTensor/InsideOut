import os
import json
import argparse
import re
import torch
from tqdm import tqdm
import csv
from PersonaMem.prepare_blocks import *
from src.llm import Agent


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
                "你是一个决策模块，从候选选项中选择最符合用户个性化问题的回答。\n\n"
                "你将看到：\n"
                "1. 用户问题；\n"
                "2. 若干候选回答选项；\n"
                "你的目标是：**为这个用户的问题选出最合适的一条选项回答**。\n\n"
                "请严格遵循以下规则进行选择：\n"
                "- 不要编造新的选项，不要修改候选选项的内容；\n"
                "- 不要输出任何分析过程或解释，只输出最终答案；\n"
                "- 最终答案格式必须为：<final_answer>(x)，其中 x ∈ {a, b, c, d}。"
            )
            
        prompt = (
            f"{instructions}\n\n"
            f"用户问题：\n{question}\n\n"
            f"候选回答选项:\n{all_options}"
        )

        response, success = self.agent.run(prompt)

        print("model prompt:", prompt)
        print("model success:", success)
        print("model response:", response)

        return response



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


def run_evaluation(cmd_args, llm):
    question_path = cmd_args.question_path
    result_path = cmd_args.result_path

    if os.path.exists(result_path):
        os.remove(result_path)

    all_errors = []
    total_rows = count_csv_rows(question_path)

    for row_data in tqdm(load_rows_with_context(question_path), total=total_rows):
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
            
            # Send the query to the LLM
            model_response = llm.query_llm(question, all_options)
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
    parser.add_argument('--result_path', type=str, default='results/Onlyllm/eval_results_32k.csv', help='Path to save the results CSV file')

    cmd_args = parser.parse_args()

    llm = Evaluation()

    run_evaluation(cmd_args, llm)

# nohup python -u infer_llm.py  > results/Onlyllm/eval_results_32k.log 2>&1 &

