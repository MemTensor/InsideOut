import json
import re
from copy import deepcopy
from typing import Any, Dict

def keep_leaf_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归清洗字典：
    - 保留叶子节点为非空字符串的项；
    - 删除空值；
    - 保留有效叶子的路径。
    """
    # 如果是字符串（叶子）
    if isinstance(d, str):
        return d if d.strip() != "" else None
    # 如果是字典
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            cleaned = keep_leaf_keys(value)
            if cleaned is not None:   # 只保留有效项
                new_dict[key] = cleaned

        # 如果这个层级所有子项都被删除，则返回 None
        return new_dict if new_dict else None
    return None

class Prompt:
    """
    A universal prompt builder for LLM-based attribute extraction.
    Designed to work with any LLM (ChatGPT, DeepSeek, Claude, Qwen, Llama, etc.).
    """

    def __init__(self, formatted_str):
        """
        初始化配置
        :param formatted_str: 预格式化的JSON字符串
        """
        self.PERSONA_SCHEMA_JSON = formatted_str.strip()

        
        
    def attribute_extraction(self, user_input: str) -> str:
        """
        Build a prompt that instructs an LLM to extract user attributes
        and rewrite them into a concise descriptive paragraph.
        """
        attribute_text=json.loads(self.PERSONA_SCHEMA_JSON)
        attribute_text=keep_leaf_keys(attribute_text)
        schema=json.dumps(attribute_text, ensure_ascii=False,indent=4)
        
        return f"""你是一个专门从对话中抽取用户画像的模块。

给定以下信息：
(1) 对话历史；
(2) 从之前对话中结构化提取出的用户属性 schema。

你的目标：从对话历史中**完整且不遗漏地**提取所有与“用户”相关的事实信息和个性化特征，并将其改写为**一个描述性段落**。

### 特别说明（关于 "system" 信息）：
- 对话历史中可能包含一条或多条 "system" 消息，用于说明当前对话的用户是谁、提供背景设定或额外说明；
- 这些 "system" 消息**只用于帮助你理解这个用户和对话上下文**；
- **不要直接从 "system" 消息中抽取或加入任何新的用户属性**，你的信息来源应当只限于 "user" 和 "assistant" 的对话内容本身。

### 关于 schema：
- 下方的 schema 中给出了**已经成功结构化的用户属性信息**；
- 请将 schema 视为“已记录的信息”，**不要重复抽取、罗列或简单改写其中已经存在的字段**；
- 只有当对话中出现了 schema 尚未覆盖的额外事实、细节或偏好，才需要在画像段落中体现；
- 如对话内容与 schema 存在冲突，以对话中的**最新明确表述**为准。


### 说明：
1. 对提取到的信息进行自然的合并和改写，使其成为一个连贯的段落：
   - 可以用多句话来覆盖全部信息，但整体应构成一个自然段落；
   - 避免简单逐条罗列原句，要用自然语言总结；
   - 如果某一信息在对话中多次出现，整合为一致的描述，而不是重复。
   - 你可以利用 "system" 中给出的身份指示，来帮助你用合适的指代方式，但不要从中新增事实。

2. 输出格式要求：
   - 只输出**最终生成的一个段落**；
   - 不要使用项目符号列表，不要加标题，不要添加任何额外解释说明或前后缀标记。


### 对话历史：
{user_input}


### 人物画像模式：
{schema}

### 输出：
"""

    def attribute_extraction_system(self, user_input: str) -> str:
        """
        Build a prompt that instructs an LLM to extract user attributes
        and rewrite them into a concise descriptive paragraph.
        """
        return f"""你是一个专门从对话中抽取用户画像的模块。

给定以下对话历史，你需要从中**完整且不遗漏地**提取所有与“用户”相关的事实信息和个性化特征，并将其改写为**一个精炼但信息尽可能全面的描述性段落**。

### 你的任务：

1. 从对话中穷尽式提取所有明确陈述或被强烈暗示的个人信息与个性化特征。

2. **任何在对话中出现过、与用户本人有关且有助于刻画其个性或习惯的细节，都不要遗漏**；但绝对不要凭空编造或过度推断，凡未被提及的内容，一律省略。

3. 对提取到的信息进行自然的合并和改写，使其成为一个连贯的段落：
   - 可以用多句话来覆盖全部信息，但整体应构成一个自然段落；
   - 避免简单逐条罗列原句，要用自然语言总结；
   - 如果某一信息在对话中多次出现，整合为一致的描述，而不是重复。

4. 输出格式要求：
   - 只输出**最终生成的一个段落**；
   - 不要使用项目符号列表，不要加标题，不要添加任何额外解释说明或前后缀标记。

### 对话历史：
{user_input}


### 输出：
"""

    def attribute_to_tree_ops(self, attribute_text: str) -> str:
        """
        Build a prompt for mapping an attribute paragraph into tree operations.
        Output operations include ADD / UPDATE / DELETE / NO_OP.
        """
        schema = self.PERSONA_SCHEMA_JSON

        return f"""你是一个记忆树操作生成器。

你将获得：
(1) 一个初始的人物画像模式，以分层 JSON 树的形式表示。
(2) 一段用自然语言撰写的用户个人属性总结段落。

你的目标是：将该段落转换为一系列用于更新该人物画像模式的操作列表，**尽可能完整地覆盖其中所有关于这个人的信息，尤其是个性化特征**。

### 关于 schema：
- 下方的 schema 中给出了**已经成功结构化的用户属性信息**；
- 请将 schema 视为“已记录的信息”，**不要重复抽取已经存在的字段**；
- 只有当人物属性段落中出现了 schema 尚未覆盖的额外事实、细节或偏好，才需要对 schema 生成操作；
- 如人物属性段落与 schema 存在冲突，以段落中的**最新明确表述**为准。

### 关于 ADD / UPDATE / DELETE / NO_OP 的使用原则：
   * 当某个属性在该路径之前**完全没有记录**时，使用：ADD(path, "value")。鼓励生成更多的分支，避免单个属性内容过长。
   * 当某个属性在该路径已有记录，本段落是对其的**补充、细化或更正**时，使用：UPDATE(path, "value")。
   * 只有在段落中明确表示某个原有信息**不再成立、被否定或需要移除**时，才使用：DELETE(path, None)。
   * 若段落中完全没有涉及到改变任何内容，只输出一行 NO_OP()。

### 关于 UPDATE 时 “value” 的关键要求（非常重要）：
   * “value” 必须在语义上**包含或融合原来的有效信息，同时加入或反映新的信息**，形成一个更完整、更准确的最新描述。
   * **绝对禁止**在 UPDATE 时仅保留新信息而丢弃原来的有用内容。
   * 当新信息是补充或细化时，value 应该是“原信息 + 新补充”的综合表述；
   * 当新信息与旧信息存在冲突时，value 应当描述“当前最新、最合理的状态”，但在可能的情况下仍要保留那些不冲突的旧细节。

### 说明：

1. 将 JSON 模式中的每一个叶子节点视为可以存储文本值的属性槽。

2. 对于段落中提到的每一项不同的用户个人属性：
   * 在模式中找到与之最匹配、且最具体的叶子节点。
   * 为该属性**生成且只生成一个**操作。

3. 你只能使用以下几种操作：
   * ADD(path, "value")
   * UPDATE(path, "value")
   * DELETE(path, None)
   * NO_OP()

4. “path” 的格式要求：
   * 使用以英文句点分隔的 JSON 键路径。
   * 示例：
     1_Biological_Characteristics.Physiological_Status.Age_Related_Characteristics.Chronological_Age

5. “value” 的格式要求：
   * 为从段落中抽取或规范化后的自然语言表述。
   * 必须用英文双引号括起来。

6. 输出格式（必须严格遵守）：
   * 只输出操作，每行一个操作。
   * 不要添加任何解释或注释。
   * 唯一允许的形式是：
     ADD(<path>, "<value>")
     UPDATE(<path>, "<value>")
     DELETE(<path>, None)
     NO_OP()

### 示例（不要照抄此示例；仅作说明用）：

# 示例段落：
# “Mary is 24 years old. She is outgoing and habitually wears a beret.”

# 示例正确操作：
# UPDATE(4_Identity_Characteristics.Social_Identity.Legal_and_Civic_Status.Name, "Mary")
# UPDATE(1_Biological_Characteristics.Physiological_Status.Age_Related_Characteristics.Chronological_Age, "24 years old")
# UPDATE(3_Personality_Characteristics.Core_Personality.Extraversion, "outgoing")
# UPDATE(5_Behavioral_Characteristics.Behavioral_Habits.Clothing_Habits, "habitually wears a beret")


### 人物画像模式：
{schema}


### 人物属性段落：
{attribute_text}


现在，请根据给定的属性段落，仅输出操作：
"""
    
    def dialogue_to_tree_ops(self, dialogue_text: str) -> str:
        """
        Build a prompt for mapping an attribute paragraph into tree operations.
        Output operations include ADD / UPDATE / DELETE / NO_OP.
        """
        schema = self.PERSONA_SCHEMA_JSON

        return f"""你是一个记忆树操作生成器。

你将获得：
(1) 一个初始的人物画像模式，以分层 JSON 树的形式表示。
(2) 一段对话历史。

你的目标是：将该对话历史转换为一系列用于更新该人物画像模式的操作列表，**尽可能完整地覆盖其中所有关于这个人的信息，尤其是个性化特征**。

### 关于 schema：
- 下方的 schema 中给出了**已经成功结构化的用户属性信息**；
- 请将 schema 视为“已记录的信息”，**不要重复抽取已经存在的字段**；
- 只有当对话历史中出现了 schema 尚未覆盖的额外事实、细节或偏好，才需要对 schema 生成操作；
- 如对话历史与 schema 存在冲突，以段落中的**最新明确表述**为准。

### 关于 ADD / UPDATE / DELETE / NO_OP 的使用原则：
   * 当某个属性在该路径之前**完全没有记录**时，使用：ADD(path, "value")。鼓励生成更多的分支，避免单个属性内容过长。
   * 当某个属性在该路径已有记录，本段落是对其的**补充、细化或更正**时，使用：UPDATE(path, "value")。
   * 只有在段落中明确表示某个原有信息**不再成立、被否定或需要移除**时，才使用：DELETE(path, None)。
   * 若段落中完全没有涉及到改变任何内容，只输出一行 NO_OP()。

### 关于 UPDATE 时 “value” 的关键要求（非常重要）：
   * “value” 必须在语义上**包含或融合原来的有效信息，同时加入或反映新的信息**，形成一个更完整、更准确的最新描述。
   * **绝对禁止**在 UPDATE 时仅保留新信息而丢弃原来的有用内容。
   * 当新信息是补充或细化时，value 应该是“原信息 + 新补充”的综合表述；
   * 当新信息与旧信息存在冲突时，value 应当描述“当前最新、最合理的状态”，但在可能的情况下仍要保留那些不冲突的旧细节。

### 说明：

1. 将 JSON 模式中的每一个叶子节点视为可以存储文本值的属性槽。

2. 对于对话历史中提到的每一项不同的用户个人属性：
   * 在模式中找到与之最匹配、且最具体的叶子节点。
   * 为该属性**生成且只生成一个**操作。

3. 你只能使用以下几种操作：
   * ADD(path, "value")
   * UPDATE(path, "value")
   * DELETE(path, None)
   * NO_OP()

4. “path” 的格式要求：
   * 使用以英文句点分隔的 JSON 键路径。
   * 示例：
     1_Biological_Characteristics.Physiological_Status.Age_Related_Characteristics.Chronological_Age

5. “value” 的格式要求：
   * 为从对话历史中抽取或规范化后的自然语言表述。
   * 必须用英文双引号括起来。

6. 输出格式（必须严格遵守）：
   * 只输出操作，每行一个操作。
   * 不要添加任何解释或注释。
   * 唯一允许的形式是：
     ADD(<path>, "<value>")
     UPDATE(<path>, "<value>")
     DELETE(<path>, None)
     NO_OP()

### 示例（不要照抄此示例；仅作说明用）：

# 示例正确操作：
# UPDATE(4_Identity_Characteristics.Social_Identity.Legal_and_Civic_Status.Name, "Mary")
# UPDATE(1_Biological_Characteristics.Physiological_Status.Age_Related_Characteristics.Chronological_Age, "24 years old")
# UPDATE(3_Personality_Characteristics.Core_Personality.Extraversion, "outgoing")
# UPDATE(5_Behavioral_Characteristics.Behavioral_Habits.Clothing_Habits, "habitually wears a beret")


### 人物画像模式：
{schema}


### 对话历史：
{dialogue_text}


现在，请根据给定的对话历史，仅输出操作：
"""




def apply_ops_to_tree(
    base_schema_json: str,
    ops_text: str,
    delete_sets_empty: bool = True,
):
    """
    Apply a list of operations produced by the LLM to a persona tree.

    Parameters
    ----------
    base_schema_json : str
        The JSON string defining the base persona schema.
        This function will NOT modify the original schema; it returns a new dict.

    ops_text : str
        Raw LLM output containing operations such as:
            ADD(path, "value")
            UPDATE(path, "value")
            DELETE(path, None)
            NO_OP()

    delete_sets_empty : bool
        If True, DELETE() sets the leaf to an empty string "".
        Otherwise DELETE() sets the leaf to None.

    Returns
    -------
    updated_tree : dict
        A Python dictionary representing the updated persona tree.
    """

    # Load schema into a Python dict and clone it
    base_tree = json.loads(base_schema_json)
    tree = deepcopy(base_tree)

    # Regex patterns for supported operations
    re_add = re.compile(r'^ADD\(([^,]+),\s*"(.*)"\)\s*$', re.DOTALL)
    re_update = re.compile(r'^UPDATE\(([^,]+),\s*"(.*)"\)\s*$', re.DOTALL)
    re_delete = re.compile(r'^DELETE\(([^,]+),\s*None\)\s*$')
    re_noop = re.compile(r'^NO_OP\(\)\s*$')

    def ensure_parent_node(t: dict, keys):
        """
        Ensure all parent nodes exist in the tree.
        For a path A.B.C.D, ensure A → B → C exist and return the dict at C.
        """
        cur = t
        for k in keys[:-1]:
            k = k.strip()
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        return cur

    # Process each line of LLM output
    for raw_line in ops_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # NO_OP → skip
        if re_noop.match(line):
            continue

        # ADD or UPDATE
        m = re_add.match(line) or re_update.match(line)
        if m:
            path_str, value = m.groups()
            path_str = path_str.strip()
            keys = [k.strip() for k in path_str.split(".")]

            parent = ensure_parent_node(tree, keys)
            last_key = keys[-1]
            parent[last_key] = value
            continue

        # DELETE
        m = re_delete.match(line)
        if m:
            path_str = m.group(1).strip()
            keys = [k.strip() for k in path_str.split(".")]

            parent = ensure_parent_node(tree, keys)
            last_key = keys[-1]
            parent[last_key] = "" if delete_sets_empty else None
            continue

        # Unrecognized line → warn
        print(f"[apply_ops_to_tree] Warning: cannot parse op line: {line!r}")

    return tree