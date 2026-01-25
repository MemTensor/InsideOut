import json
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import os
from src.segment import split_conversation
from src.construct import run_full_pipeline

def load_jsonl_to_dict(filepath):
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue 
            obj = json.loads(line)

            for k, v in obj.items():
                result[k] = v
    return result

# ------------------------------------------------------------------------------------
# 并行操作

def process_one_key(item):
    """
    item: (key, value)
    返回: (key, tmp_data, final_tree)
    这里只处理单个 key 下的所有 chunk，内部保持串行。
    """
    key, value = item

    print(f"Key: {key}")
    print("-" * 50)

    with open('src/construct/human_tree_en.json', 'r', encoding='utf-8') as file:
        memtree = json.load(file)
    memtree = json.dumps(memtree, indent=4, ensure_ascii=False).strip()

    tmp_data = {}
    final_chunks = split_conversation(value)  #,max_pairs_per_chunk=3
    tmp_data['final_chunks'] = final_chunks

    tmp = []
    for chunk in tqdm(final_chunks, desc=f"{key} chunks", leave=False):
        mem_all_data = run_full_pipeline(chunk, memtree)
        memtree = json.dumps(mem_all_data['updated_tree'], indent=4, ensure_ascii=False).strip()
        tmp.append(mem_all_data)

    tmp_data['mem_all_data'] = tmp
    final_tree = json.loads(memtree)

    return key, tmp_data, final_tree

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="例如：data/shared_contexts_32k.jsonl"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/DeepSeek-V3",
        help="输出的根目录"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="并行进程数"
    )
    args = parser.parse_args()
    
    filepath = args.filepath
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_dir = os.path.join(args.output_root, base_name)  
    os.makedirs(output_dir, exist_ok=True)
    
    data_dict = load_jsonl_to_dict(filepath)

    all_tmp_data={}
    all_final_tree={}
    all_tmp_data_path = os.path.join(output_dir, "all_tmp_data.json")
    all_final_tree_path = os.path.join(output_dir, "all_final_tree.json")
    
    num_workers = args.num_workers

    items = list(data_dict.items())

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_key = {
            executor.submit(process_one_key, item): item[0]
            for item in items
        }

        for future in tqdm(as_completed(future_to_key),
                        total=len(future_to_key),
                        desc="processing keys"):
            key = future_to_key[future]
            try:
                key_ret, tmp_data, final_tree = future.result()
            except Exception as e:
                print(f"处理 {key} 时出错: {e}")
                continue

            all_tmp_data[key_ret] = tmp_data
            all_final_tree[key_ret] = final_tree

            with open(all_tmp_data_path, "w", encoding="utf-8") as f:
                json.dump(all_tmp_data, f, ensure_ascii=False, indent=2)

            with open(all_final_tree_path, "w", encoding="utf-8") as f:
                json.dump(all_final_tree, f, ensure_ascii=False, indent=2)

            print(f"保存完成：{key_ret}")
    
# nohup python -u pipline_ablation.py --num_workers 5 --filepath data/shared_contexts_32k.jsonl --output_root data/longcat-flash-chat > data/longcat-flash-chat/shared_contexts_32k.log 2>&1 &

