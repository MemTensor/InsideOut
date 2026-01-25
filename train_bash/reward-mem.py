import re
import json
import ast
import traceback
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Optional

import json_repair
from swift.plugin import ORM, orms
from src.llm import Agent
from openai import OpenAI


PROMPT = """你是一名严格的“属性树操作整体评分器”。你的任务是：根据真实标注操作序列 GT_Ops，对模型预测操作序列 Pred_Ops 的整体质量打一个分数 score ∈ [-1, 1]。

【输入】
- GT_Ops（真实标注）：操作序列列表，元素形式如 ADD(path, value) / UPDATE(path, value) / DELETE(path, value) / NO_OP()
- Pred_Ops（模型预测）：操作序列列表，格式同上


【重要约束】
1) 你只输出一个 JSON：{{"score": <float>}}，不要输出任何解释、不要输出多余字段。
2) score 必须是 [-1, 1] 之间的“连续浮点数”（可取任意值），建议保留 2 位小数。
3) 下方“分数档位参考”仅作为锚点用于对齐整体质量，你需要在锚点之间进行微调，输出更精细的分数。
4) 例如整体介于 0.7 与 1.0 之间，就输出 0.71~0.99 的某个值；介于 0.5 与 0.7 之间，就输出 0.51~0.69 的某个值；以此类推。


【分数档位参考（整体质量锚点）】
* 1.0（几乎完美）：Pred 与 GT 在关键操作上几乎完全一致；type/path 几乎一致；value 语义等价；无多余操作。
* 0.7（高质量）：关键操作大部分正确；仅少量 value 细节偏差，或极少量缺失/冗余。
* 0.5（中等可用）：整体思路与核心方向正确；存在少量缺失/冗余；部分 path/value 错误，但不影响主要语义。
* 0.3（部分可靠）：约一半内容可靠；关键操作有对有错，需要一些修正。
* 0.0（少量正确）：仅少量操作或片段正确；缺失/冗余与错误较明显；关键操作有对有错。
* -0.3（勉强相关）：大体相关但缺失/错误较多；只能看出在尝试完成任务，基本不可直接使用。
* -0.5（明显偏离）：多数关键操作缺失或错误；较多错 path/type 或明显多余操作，整体偏离预期。
* -0.7（灾难性）：结构/语义大范围错乱，几乎不可用。
* -1.0（无意义输出）：明显无意义、垃圾文本或与任务无关。


【输出格式】
仅输出包含评分结果的 JSON 对象，不要输出任何额外说明和解释的内容。
只输出：
{{"score": <float>}}


【任务数据】
  - GT_Ops:
  {gt_ops}

  - Pred_Ops:
  {pred_ops}
"""





class TreeOpRewardFunction(ORM):
    """
    Reward aligned with 奖励函数.docx:
    - For each GT op, find best matching pred op (one-to-one matching).
    - EXACT: +1.0
    - PARTIAL: ~0.5 (float in (0,1))
    - MISSING: -0.5
    - WRONG: -1.0
    - Redundancy penalty when pred has明显冗余, treat as an extra scoring term.
    - Final clip to [-1,1].
    """

    def _clip(self, x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))


    def _extract_operations(self, text: str) -> list:
        pattern = re.compile(
            r"""
            (?:ADD|UPDATE|DELETE)      # operation type
            \(
                [^,()]+               # path
                \s*,\s*
                "(?:\\.|[^"])*"       # quoted value
            \)
            |
            NO_OP\(\)
            """,
            re.VERBOSE
        )
        return pattern.findall(text)
    

    def _redundancy_penalty(self, n_gt: int, n_pred: int) -> float:
        """
        冗余惩罚项：当 pred 明显多于 gt 时，引入 penalty，并作为额外评分项参与平均。
        返回范围 [-1, 0]
        """
        if n_pred <= n_gt:
            return 0.0
        extra = n_pred - n_gt
        pen = - float(extra) / float(max(n_gt, 1))
        return max(-1.0, pen)


    def score_pair(self, pred_text: str, gt_text: str) -> float:
        gt_ops = gt_text
        pred_ops = pred_text

        try:
            agent = Agent("You are a helpful assistant.")
            response, success = agent.run(
                PROMPT.format(
                    gt_ops=gt_ops, 
                    pred_ops=pred_ops
                )
            )
            
            if success:
                try:
                  m = re.search(r"\{.*\}", response, flags=re.S)
                  final=float(json.loads(m.group(0))["score"])
                except:
                  final=0
              
        except Exception as e:
            err = traceback.format_exc()
            print(err)
            final=0

        return self._clip(final, -1.0, 1.0)

    
    def _unpack_and_score(self, args):
        """
        辅助方法：解包参数并执行评分。
        用于 executor.map，因为它只接受一个参数。
        """
        pred, gt = args
        try:
            return float(self.score_pair(pred_text=pred, gt_text=gt))
        except Exception as e:
            # 返回保底分
            print(f"Error in thread processing: {e}")
            return 0


    def __call__(self, completions, solution, **kwargs):
        data_pairs = list(zip(completions, solution))
        max_workers = 10
        print("生成结果和真值结果：",data_pairs)
        rewards = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map 会保持结果顺序与输入顺序一致
            results = executor.map(self._unpack_and_score, data_pairs)
            rewards = list(results)
        print(rewards)
        return rewards



orms["tree_op_reward"] = TreeOpRewardFunction


if __name__ == "__main__":
    import time 
    
    # 初始化
    reward_fn = TreeOpRewardFunction()

    # --- 测试用例 1: 完美匹配 ---
    gt_1 = """
    ADD(4_Identity.Name, "Kanoa Manu")
    UPDATE(4_Identity.Job, "Senior Software Engineer")
    """
    pred_1 = """
    ADD(4_Identity.Name, "Kanoa Manu")
    UPDATE(4_Identity.Job, "Senior Software Engineer")
    """
    
    score_1 = reward_fn.score_pair(pred_1, gt_1)
    print(f"Test Case 1 (Perfect Match): Expected ~1.0, Got: {score_1}")


    # --- 测试用例 2: 冗余惩罚 ---
    gt_2 = 'ADD(Path.To.Key, "OnlyGT")'
    pred_2 = """
    ADD(Path.To.Key, "OnlyGT")
    ADD(Path.To.Garbage, "Redundant info")
    """
    
    score_2 = reward_fn.score_pair(pred_2, gt_2)
    print(f"Test Case 2 (Redundancy): Expected ~0.0, Got: {score_2}")


    # --- 测试用例 3: 混合示例 ---
    gt_3 = '''ADD(4_Identity_Characteristics.Social_Identity.Legal_and_Civic_Status.Name, "Kanoa Manu")
    UPDATE(4_Identity_Characteristics.Motivations_and_Goals.Goals.Long_Term, "Create a music fusion app")
    DELETE(2_Social_Connections.Old_Relationships.Ex_Partner, "Sarah")
    ADD(1_Biological_Characteristics.Physiological_Status.Age_Related_Characteristics.Chronological_Age, "32")'''
    pred_3 = """ADD(4_Identity_Characteristics.Social_Identity.Legal_and_Civic_Status.Name, "Mr. Kanoa Manu")
    ADD(4_Identity_Characteristics.Motivations_and_Goals.Goals.Long_Term, "Build an app for music")
    DELETE(2_Social_Connections.Old_Relationships.Ex_Partner, "Sarah")
    """
    
    score_3 = reward_fn.score_pair(pred_3, gt_3)
    print(f"Test Case 3 (Hybrid): Expected ~0.0, Got: {score_3}")

    # --- 测试用例 4: 多线程示例 ---
    # --- 构造数据 ---
    # 构造 3 条数据
    completions = [pred_1, pred_2, pred_3]
    solution = [gt_1, gt_2, pred_3]

    start_time = time.time()

    # --- 执行调用 ---
    rewards = reward_fn(completions, solution)

    end_time = time.time()
    duration = end_time - start_time

    # --- 结果分析 ---
    print(f"\nRewards: {rewards}")
    print(f"Total Time: {duration:.2f} seconds")