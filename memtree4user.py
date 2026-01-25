
import json
import re
from datetime import datetime
from pymongo import MongoClient
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.segment import split_conversation
from src.construct import run_full_pipeline_1

import faulthandler
#faulthandler.enable()
#faulthandler.dump_traceback_later(20, repeat=True)


_MONGO_CLIENT = None
_TREE_COL = None

def get_tree_col():
    """Each process will create its own MongoClient once."""
    global _MONGO_CLIENT, _TREE_COL
    if _TREE_COL is None:
        uri = os.getenv("MONGO_URI", "mongodb://ip:port")
        _MONGO_CLIENT = MongoClient(uri)
        db = _MONGO_CLIENT["persona_mem"]
        _TREE_COL = db["attribute_trees"]
    return _TREE_COL




def load_initial_tree():
    """从 human_tree_en.json 载入初始属性树"""
    with open('src/construct/human_tree_en.json', 'r', encoding='utf-8') as f:
        return json.load(f)


# === 2. 给 LLM 流水线用的“整棵树读写” ===

def get_or_init_tree(user_id: str):
    """
    从 Mongo 中读取某用户的整棵属性树。
    若不存在，则用 initial_tree 初始化后返回。
    """
    tree_col = get_tree_col()
    doc = tree_col.find_one({"user_id": user_id}, {"_id": 0, "tree": 1})
    if doc and "tree" in doc:
        return doc["tree"]

    # 不存在，插入一棵初始树
    tree = load_initial_tree()
    tree_col.update_one(
        {"user_id": user_id},
        {
            "$setOnInsert": {
                "tree": tree,
                "created_at": datetime.utcnow(),
            },
            "$set": {
                "updated_at": datetime.utcnow(),
            },
        },
        upsert=True,
    )
    return tree


def save_full_tree(user_id: str, tree: dict):
    """把整棵属性树写回 Mongo（LLM 流水线跑完后用）"""
    tree_col = get_tree_col()
    tree_col.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "tree": tree,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


# === 3. 给“直接操作属性树（UPDATE/ADD/DELETE）”用的局部更新 ===

def ensure_doc_exists(user_id: str):
    """
    确保用户的文档存在，但不一定加载整棵树。
    如果你希望新用户也有完整 initial_tree，可以在这里 set 初始树；
    如果不强制，可以只插一个空 tree。
    """
    # 这里做一个“没有就插空 tree”的策略，你也可以换成 load_initial_tree()
    tree_col = get_tree_col()
    tree_col.update_one(
        {"user_id": user_id},
        {
            "$setOnInsert": {
                "tree": {},
                "created_at": datetime.utcnow(),
            },
            "$set": {
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


def update_path(user_id: str, path: str, value):
    """
    对 tree.<path> 做 UPDATE，不读整棵树：
       UPDATE(5_Behavioral_Characteristics.Behavioral_Habits.Clothing_Habits, "xxx")
    对应 Mongo: {$set: {"tree.5_Behavioral_Characteristics.Behavioral_Habits.Clothing_Habits": "xxx"}}
    """
    ensure_doc_exists(user_id)
    mongo_path = f"tree.{path}"
    tree_col = get_tree_col()
    tree_col.update_one(
        {"user_id": user_id},
        {
            "$set": {
                mongo_path: value,
                "updated_at": datetime.utcnow(),
            }
        }
    )


def delete_path(user_id: str, path: str):
    """DELETE(path) -> 对 tree.<path> 做 unset"""
    ensure_doc_exists(user_id)
    mongo_path = f"tree.{path}"
    tree_col = get_tree_col()
    tree_col.update_one(
        {"user_id": user_id},
        {
            "$unset": {mongo_path: ""},
            "$set": {"updated_at": datetime.utcnow()},
        }
    )


# -------------------------/tree_ops_parser.py
# 支持 UPDATE(path, "value") / DELETE(path)
UPDATE_RE = re.compile(
    r'^UPDATE\(\s*([A-Za-z0-9_\.]+)\s*,\s*"(.*)"\s*\)\s*$'
)
DELETE_RE = re.compile(
    r'^DELETE\(\s*([A-Za-z0-9_\.]+)\s*\)\s*$'
)


def parse_op(op_str: str):
    """
    解析一条操作字符串，返回 (op_type, path, value)
    op_type: "UPDATE" / "DELETE"
    """
    s = op_str.strip()

    m = UPDATE_RE.match(s)
    if m:
        path, value = m.group(1), m.group(2)
        return "UPDATE", path, value

    m = DELETE_RE.match(s)
    if m:
        path = m.group(1)
        return "DELETE", path, None

    raise ValueError(f"无法解析操作: {op_str}")



# -------------------------/tree_ops_executor.py
def apply_single_op(user_id: str, op_str: str):
    """
    对某个用户执行一条树操作（UPDATE/DELETE）。
    不会把整个树读出来。
    """
    op_type, path, value = parse_op(op_str)

    if op_type == "UPDATE":
        update_path(user_id, path, value)
    elif op_type == "DELETE":
        delete_path(user_id, path)
    else:
        raise ValueError(f"暂不支持的 op_type: {op_type}")



# -------------------------处理对话历史文件
def process_one_key(item):
    """
    item: (key, value) 这里的 key 可以理解为 user_id
    返回: (key, tmp_data, final_tree)
    """
    key, value = item

    print(f"Key: {key}")
    print("-" * 50)

    # 1. 从 Mongo 获取/初始化该用户的属性树（整棵树）
    memtree = get_or_init_tree(key)     # dict
    memtree = json.dumps(memtree, ensure_ascii=False).strip()  # run_full_pipeline_1 还是吃 string

    tmp_data = {}
    final_chunks = split_conversation(value)
    tmp_data['final_chunks'] = final_chunks

    tmp = []
    for chunk in tqdm(final_chunks, desc=f"{key} chunks", leave=False):
        #print(f"[DEBUG] entering run_full_pipeline_1", flush=True)
        mem_all_data = run_full_pipeline_1(chunk, memtree)
        #print(f"[DEBUG] returned from run_full_pipeline_1", flush=True)
        # updated_tree 通常是 dict，这里统一转 string 继续传给下一个 chunk
        memtree = json.dumps(mem_all_data['updated_tree'], ensure_ascii=False).strip()
        tmp.append(mem_all_data)

    tmp_data['mem_all_data'] = tmp
    final_tree = json.loads(memtree)

    # 2. 把最终树写回 Mongo
    save_full_tree(key, final_tree)

    return key, tmp_data, final_tree


# -------------------------处理一小段对话
def process_single_dialogue(user_id: str, dialogue_text: str):
    """
    处理单个用户的一段对话，更新该用户在 Mongo 中的属性树。
    """
    memtree = get_or_init_tree(user_id)
    memtree_str = json.dumps(memtree, ensure_ascii=False).strip()

    chunks = split_conversation(dialogue_text)
    tmp = []
    for chunk in chunks:
        mem_all_data = run_full_pipeline_1(chunk, memtree_str)
        memtree_str = json.dumps(mem_all_data['updated_tree'], ensure_ascii=False).strip()
        tmp.append(mem_all_data)

    final_tree = json.loads(memtree_str)
    save_full_tree(user_id, final_tree)
    return final_tree, tmp




def load_jsonl_to_dict(filepath):
    result = {}
    # 逐行读取
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 每行是一个 JSON 对象
            obj = json.loads(line)

            # 合并到最终字典中
            # 每行 JSON 都是 {key: value} 结构
            for k, v in obj.items():
                result[k] = v
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="file",
        choices=["file", "dialogue", "op"],
        help="file: 处理对话历史文件; dialogue: 单条对话; op: 直接操作属性树"
    )
    parser.add_argument(
        "--filepath",
        type=str,
        help="mode=file 时：例如 data/shared_contexts_32k.jsonl"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/DeepSeek-V3",
        help="输出的根目录（仅 mode=file 用到）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="并行进程数（mode=file 时）"
    )
    parser.add_argument(
        "--user_id",
        type=str,
        help="mode=dialogue/op 时需要指定 user_id（对应原来的 key）"
    )
    parser.add_argument(
        "--dialogue",
        type=str,
        help="mode=dialogue 时：传入一段对话文本（或你可以自己扩展成文件路径）"
    )
    parser.add_argument(
        "--op",
        type=str,
        help='mode=op 时：例如 UPDATE(5_Behavioral_Characteristics.Behavioral_Habits.Clothing_Habits, "habitually wears a beret")'
    )
    args = parser.parse_args()

    if args.mode == "file":
        # === 保留你原来的批处理逻辑，只是用新的 process_one_key ===
        filepath = args.filepath
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        output_dir = os.path.join(args.output_root, base_name)  
        os.makedirs(output_dir, exist_ok=True)
        
        data_dict = load_jsonl_to_dict(filepath)

        all_tmp_data = {}
        all_final_tree = {}
        all_tmp_data_path = os.path.join(output_dir, "all_tmp_data.json")
        all_final_tree_path = os.path.join(output_dir, "all_final_tree.json")
        
        num_workers = args.num_workers
        items = list(data_dict.items())

        if num_workers == 1:
            for item in tqdm(items, desc="processing keys"):
                key = item[0]
                try:
                    key_ret, tmp_data, final_tree = process_one_key(item)
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
        else:

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

    elif args.mode == "dialogue":
        final_tree, _ = process_single_dialogue(args.user_id, args.dialogue)
        print(json.dumps(final_tree, ensure_ascii=False, indent=2))

    elif args.mode == "op":
        apply_single_op(args.user_id, args.op)
        print(f"已对用户 {args.user_id} 执行操作: {args.op}")
