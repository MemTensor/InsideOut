
def split_conversation(
    messages,
    max_pairs_per_chunk=10,   # 每个小块最多多少对 user/assistant
    max_len=8000             # 每块最大字符长度
):
    """
    输入: messages = [{"role": "...", "content": "..."}, ...]
    输出: 一个大列表，每个元素是一个小块(列表)
    """

    # 1. 先按 system 分段，形成按系统提示分开的大段
    system_blocks = []
    current_block = []

    for msg in messages:
        # 忽略不符合预期结构的东西（比如混进来的非 dict）
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue

        if msg["role"] == "system":
            if current_block:
                system_blocks.append(current_block)
            # 新开一个 block，以当前 system 开头
            current_block = [msg]
        else:
            if not current_block:
                current_block = [msg]
            else:
                current_block.append(msg)
    if current_block:
        system_blocks.append(current_block)

    # 2. 在每个 system_block 里继续按 user/assistant 对 + 长度 分块
    final_chunks = []

    for block in system_blocks:
        if not block:
            continue
        if block[0]["role"] == "system":
            system_msg = block[0]
            dialog_msgs = block[1:]
            final_chunks.append([system_msg])
        else:
            system_msg = None        
            dialog_msgs = block[:]  

        # 2.1 先把 user & assistant 成对整理出来
        pairs = []
        i = 0
        while i < len(dialog_msgs):
            msg = dialog_msgs[i]
            if msg["role"] == "user":
                # 如果后一个是 assistant，就配成一对
                if i + 1 < len(dialog_msgs) and dialog_msgs[i + 1]["role"] == "assistant":
                    pairs.append((msg, dialog_msgs[i + 1]))
                    i += 2
                else:
                    # 没有 assistant 跟着，就当成单独 user 对
                    pairs.append((msg, None))
                    i += 1
            else:
                # 如果出现了裸 assistant（前面没有 user），视作单独对
                if msg["role"] == "assistant":
                    pairs.append((None, msg))
                # 其他角色就跳过
                i += 1

        # 2.2 把 pairs 按规则塞进多个 chunk
        if not pairs:
            # 如果一个 system 下啥对话都没有，也可以保留一个只含 system 的块
            continue

        def new_chunk():
            # 每个块都复用同一个 system 提示，保证每块上下文自洽
            return [system_msg.copy()]  # copy 一下避免后续意外修改
            # return []

        current_chunk = new_chunk()
        current_pairs = 0
        sys_text=system_msg["content"]
        # sys_text=''
        current_len = len(sys_text)

        for user_msg, assistant_msg in pairs:
            # 估算当前这一对的长度
            pair_len = 0
            if user_msg is not None:
                pair_len += len(user_msg["content"].split())
            if assistant_msg is not None:
                pair_len += len(assistant_msg["content"].split())

            # 如果超过对数上限 或 加上这一对会超过最大长度 -> 先收块，再开新块
            if (current_pairs >= max_pairs_per_chunk) or (current_len + pair_len > max_len):
                final_chunks.append(current_chunk)
                current_chunk = new_chunk()
                current_pairs = 0
                current_len = len(sys_text)

            # 把当前这一对塞进去
            if user_msg is not None:
                current_chunk.append(user_msg)
                current_len += len(user_msg["content"])
            if assistant_msg is not None:
                current_chunk.append(assistant_msg)
                current_len += len(assistant_msg["content"])

            current_pairs += 1

        # 把最后一个 chunk 收进去
        if current_chunk:
            final_chunks.append(current_chunk)

    return final_chunks
