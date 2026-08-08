# 导入支持 Unicode 属性（如 \p{L}）的正则表达式库，比标准 re 支持更多特性
import regex as re
# 操作系统接口，用于文件定位、获取文件大小等
import os
# pickle 序列化，用于保存/加载词汇表和合并规则，便于持久化
import pickle
# 类型提示：可迭代对象、迭代器，用于标注函数参数和返回值类型
from typing import Iterable, Iterator
# 计数器，用于统计词频和字节对频次，简化频次累加操作
from collections import Counter
# 多进程支持，利用多核 CPU 加速训练
import multiprocessing
# BinaryIO 类型提示，表示以二进制模式打开的文件对象，用于类型检查
from typing import BinaryIO
import numpy as np
# ------------------------------------------------------------
# 预编译的正则表达式，用于将文本切分为“预词元”（pre-token），模拟 GPT‑2 的分词方式。
# 该正则匹配以下模式（按顺序尝试）：
#   1. 英语常见缩写：'s, 't, 're, 've, 'm, 'll, 'd
#   2. 可选空格后跟一个或多个字母（\p{L} 匹配任意语言的字母）
#   3. 可选空格后跟一个或多个数字（\p{N} 匹配任意语言的数字）
#   4. 可选空格后跟一个既不是空格也不是字母也不是数字的字符（标点、符号等）
#   5. 连续的空白符（但不包括最后一个空格？这里的 \s+(?!\S) 匹配后面不跟非空白字符的空白，通常用于行尾）
#   6. 普通的空白符 \s+（兜底）
# 这种模式保证了文本被拆分成语义上相对独立的片段，且保留必要的空格信息。
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

# 预创建 256 个单字节对象，大幅加速字节切片时的构造开销
_SINGLE_BYTE = [bytes([i]) for i in range(256)]


# ------------------------------------------------------------
def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    在二进制文件中查找合适的切分点，使得每个切分块都以 split_special_token 作为边界。
    这样多进程处理时每个块可以独立处理，不会把一个特殊 token 切半。
    
    参数：
        file: 已打开（二进制模式）的文件对象，且处于可读状态
        desired_num_chunks: 期望的块数（通常等于 CPU 核心数）
        split_special_token: 特殊 token 的字节表示，例如 b"<|endoftext|>"
    
    返回：
        一个递增的字节偏移列表，包含 0 和 file_size，以及中间调整后的切分点。
        实际返回的块数可能少于 desired_num_chunks（如果特殊 token 稀疏）。
    """
    # 确保特殊 token 是字节串而非字符串，避免类型错误
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 将文件指针移动到末尾，获取文件总字节数
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    # 重置文件指针到开头，便于后续读取
    file.seek(0)

    # 平均每个块的大小（粗略估计，字节数）
    chunk_size = file_size // desired_num_chunks

    # 初始切分边界：等间距分割，共 desired_num_chunks+1 个点（含首尾）
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    # 最后一个边界强制设为文件末尾（避免浮点误差）
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前探测的字节数（4KB），用于搜索特殊 token

    # 对每一个内部边界（不包含第一个边界0和最后一个边界file_size）进行微调
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]   # 初始猜测位置
        file.seek(initial_position)              # 定位到该位置

        while True:
            # 从当前位置读取一个小块（最多 mini_chunk_size 字节）
            mini_chunk = file.read(mini_chunk_size)

            # 如果读到空（文件结束），说明已到文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊 token 的首次出现位置（字节偏移，相对于mini_chunk开头）
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到：边界调整为 initial_position + token 起始偏移
                chunk_boundaries[bi] = initial_position + found_at
                break

            # 没找到，继续向后移动 mini_chunk_size 字节，重新读取下一块
            initial_position += mini_chunk_size

    # 多个边界可能被调整到同一位置（例如都调整到文件末尾），去重并排序后返回
    # 注意 set 会丢失顺序，所以先 set 再 sorted
    return sorted(set(chunk_boundaries))


# ------------------------------------------------------------
def process_chunk(args: tuple[str, int, int, list[str] | None]) -> Counter:
    """
    处理一个文件块的辅助函数，由多进程调用。每个进程独立统计词频。
    
    参数 args 是一个元组，包含：
        input_path: 文件路径（字符串）
        start, end: 本块在文件中的起始和结束字节偏移（闭开区间 [start, end)）
        special_tokens: 特殊 token 列表（可能为 None），用于跳过这些 token 的统计
    
    返回：
        该块内所有“单词”（经过 PAT 切分后的字节元组）的 Counter 对象。
        元组的每个元素是一个单字节的 bytes 对象，例如 (b'c', b'a', b't')。
    """
    input_path, start, end, special_tokens = args

    # 恢复原有的按段读取，从而不会将 \n\n 这种跨行标识拆断
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    text = chunk_bytes.decode('utf-8', errors='ignore')

    if special_tokens:
        sorted_st = sorted(special_tokens, key=len, reverse=True)
        escaped_st = [re.escape(st) for st in sorted_st]
        st_pattern = "|".join(escaped_st)
        parts = re.split(f"({st_pattern})", text)
    else:
        parts = [text]

    raw_counts = Counter()
    st_set = set(special_tokens) if special_tokens else set()
    for part in parts:
        if part in st_set:
            continue
        for m in PAT.finditer(part):
            raw_counts[m.group(0)] += 1

    word_counts = Counter()
    # 仅对 unique 词进行 bytes 转换，大幅减少转换带来的时间开销
    for word, count in raw_counts.items():
        b_word = word.encode("utf-8")
        word_counts[tuple(_SINGLE_BYTE[b] for b in b_word)] += count

    return word_counts


# ------------------------------------------------------------
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。
    
    参数：
        input_path: 训练文本文件路径（UTF‑8 编码）
        vocab_size: 最终词表大小（包含单字节、特殊 token 和 BPE 合并结果）
        special_tokens: 特殊 token 列表（例如 ["<|endoftext|>"]），这些 token 不会参与 BPE 合并，
                        且会被保留为独立的词元。
    
    返回：
        vocab: 字典 {id: bytes}，id 从 0 开始，0~255 为单字节，之后依次为特殊 token 和合并结果
        merges: 合并规则列表 [(a,b), ...]，其中 a,b 为 bytes 对象，表示将 a+b 合并成一个新 token
    """
    word_counts = Counter()

    # 准备用于多进程切分的特殊 token（如果提供了 <|endoftext|>）
    # 注意：只有 <|endoftext|> 被视为天然分隔符，用于切分文件块
    endoftext_token = b"<|endoftext|>"
    has_endoftext = special_tokens and "<|endoftext|>" in special_tokens
    split_token = endoftext_token if has_endoftext else None

    # 如果可以使用 <|endoftext|> 作为分隔符，则启用多进程处理（加速训练）
    if split_token:
        # num_processes = multiprocessing.cpu_count()   # 使用所有 CPU 核心

        num_processes = min(5,multiprocessing.cpu_count())
        # 核心 OOM 优化点：不再根据 CPU 核数切分（例如导致 11GB 切成 8 份即单块 1.4GB 暴雷）
        # 我们按照绝对 50MB 大小切割！这样每份内存负担始终不超过 150MB
        file_size = os.path.getsize(input_path)
        chunk_size = 50 * 1024 * 1024
        desired_chunks = max(num_processes, file_size // chunk_size)

        with open(input_path, "rb") as f:
            # 找到按 <|endoftext|> 分隔的切分边界，使得每个边界都落在 token 起始处
            boundaries = find_chunk_boundaries(f, desired_chunks, split_token)

        # 为每个块构造参数元组 (input_path, start, end, special_tokens)
        args_list = [(input_path, boundaries[i], boundaries[i+1], special_tokens)
                     for i in range(len(boundaries)-1)]

        # 使用进程池并行处理所有块，imap_unordered 能更加平滑地减少内存峰值聚集
        with multiprocessing.Pool(processes=num_processes) as pool:
            for chunk_counts in pool.imap_unordered(process_chunk, args_list):
                word_counts.update(chunk_counts)

    else:
        # 没有合适的切分 token 时，使用自定义缓冲池分块读取，防止 f.read() 撑爆内存。
        # 同时保证缓冲块不会在连续换行符处截断（破坏 \s+ 正则的跨行语义）。
        raw_counts = Counter()
        st_set = set(special_tokens) if special_tokens else set()
        
        st_pattern_obj = None
        if special_tokens:
            sorted_st = sorted(special_tokens, key=len, reverse=True)
            escaped_st = [re.escape(st) for st in sorted_st]
            st_pattern_obj = re.compile(f"({'|'.join(escaped_st)})")

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = []
            buffer_len = 0
            chunk_limit = 50 * 1024 * 1024  # 50MB 内存上限

            for line in f:
                buffer.append(line)
                buffer_len += len(line)

                # 当累积到 50MB 时，且当前行不是纯空白（避免拦腰截断连续 \n\n ）时，进行结算
                if buffer_len >= chunk_limit and line.strip() != "":
                    text = "".join(buffer)
                    parts = st_pattern_obj.split(text) if st_pattern_obj else [text]
                    for part in parts:
                        if part in st_set:
                            continue
                        for m in PAT.finditer(part):
                            raw_counts[m.group(0)] += 1
                    buffer.clear()
                    buffer_len = 0
            
            # 收尾处理最后剩下的部分
            if buffer:
                text = "".join(buffer)
                parts = st_pattern_obj.split(text) if st_pattern_obj else [text]
                for part in parts:
                    if part in st_set:
                        continue
                    for m in PAT.finditer(part):
                        raw_counts[m.group(0)] += 1
                
        # 仅对 unique 词进行类型的转换操作，而不需要针对每个词元出现都执行转换
        for word, count in raw_counts.items():
            b_word = word.encode("utf-8")
            word_counts[tuple(_SINGLE_BYTE[b] for b in b_word)] += count

    # ---------- 初始化词表 ----------
    vocab = {i: _SINGLE_BYTE[i] for i in range(256)}   # 0~255 对应单字节
    merges = []                                   # 存储合并历史 (a, b)，a,b 为 bytes 对象

    import heapq   # 导入堆队列，用于快速获取最高频字节对

    # 定义堆节点类，用于最大堆（通过自定义 __lt__ 实现）
    class HeapNode:
        __slots__ = ['count', 'pair']   # 节省内存，限定属性
        def __init__(self, count, pair):
            self.count = count
            self.pair = pair

        def __lt__(self, other):
            # 1. 优先比较出现频次，最大堆效果：count 越大越优先（推向堆顶）
            if self.count != other.count:
                return self.count > other.count
            
            # 2. 频次相同时触发 GPT-2 的 Tie-breaker 机制：
            # 基于 tokens 的实际字节串 (bytes) 字典序降序比较。
            # 而绝不是无序的合并 ID，以保证确定的跨平台合并顺序。
            left1, right1 = self.pair
            left2, right2 = other.pair
            if left1 != left2:
                return left1 > left2  # 左节点字节串字典序大者优先
            return right1 > right2    # 右节点字节串字典序大者优先

    # 用并行数组模拟双向链表，存储每个单词的字节序列及相邻关系
    # 每个节点表示一个 token（可能是原始单字节，也可能是合并后的新 token）
    node_val = []        # 节点存储的 token（bytes 对象）
    node_prev = []       # 前一个节点的索引，-1 表示无前驱
    node_next = []       # 后一个节点的索引，-1 表示无后继
    node_valid = []      # 节点是否有效（合并过程中某些节点被标记为失效）
    node_count = []      # 该节点所属单词的频次（每个单词内所有节点共享此频次）
    
    # 用并行数组模拟单向链表，管理每个 pair 出现的位置（左节点索引）
    # 这是为了高效遍历所有出现某个字节对的位置，而不必扫描整个链表
    pair_occ_list = []       # 存储左节点索引（出现该 pair 的位置）
    pair_pre_occ_list = []   # 前一个 occurrence 索引（链表的前驱指针），-1 表示无前驱
    pair_latest_occ = {}     # 字典 {pair: 最新的 occurrence 索引}，用于快速追加新 occurrence
    pairs = Counter()        # 统计每个字节对 (a, b) 的总频次（跨所有单词）

    # 遍历所有单词，将其转换为双向链表节点，并记录所有相邻字节对
    for word, count in word_counts.items():
        if len(word) < 2:
            continue   # 长度小于2的单词无法形成任何字节对，无需处理
            
        prev_idx = -1
        for b in word:   # b 是单字节 bytes 对象
            curr_idx = len(node_val)
            node_val.append(b)
            node_prev.append(prev_idx)
            node_next.append(-1)
            node_valid.append(True)
            node_count.append(count)
            
            if prev_idx != -1:
                # 与前一节点建立连接
                node_next[prev_idx] = curr_idx
                pair = (node_val[prev_idx], b)
                pairs[pair] += count   # 增加该字节对频次
                
                # 将该位置（左节点索引）加入 occurrence 单向链表
                occ_idx = len(pair_occ_list)
                pair_occ_list.append(prev_idx)                            # 当前 occurrence 记录的位置
                pair_pre_occ_list.append(pair_latest_occ.get(pair, -1))  # 指向前一个 occurrence
                pair_latest_occ[pair] = occ_idx                           # 更新最新 occurrence 索引
                
            prev_idx = curr_idx

    # 建立堆，将所有字节对及其频次放入堆中（HeapNode 对象）
    heap = [HeapNode(count, pair) for pair, count in pairs.items()]
    heapq.heapify(heap)

    # 需要进行的合并次数 = 目标词表大小 - 256（单字节） - 特殊 token 个数
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)

    # 进行指定次数的 BPE 合并
    for _ in range(num_merges):
        best_pair = None
        # 使用惰性删除（lazy deletion）从堆中弹出合法的最高频次字节对
        # 堆中可能存有过期的条目（频次与当前 pairs 中不符），需要跳过
        while heap:
            node = heapq.heappop(heap)
            if node.count == pairs.get(node.pair, 0):
                best_pair = node.pair
                break
        
        if not best_pair:
            break   # 没有可合并的对（堆空或所有对频次为0），提前结束

        a, b = best_pair   # a, b 都是 bytes 对象
        ab = a + b          # 合并后的新字节串
        merges.append((a, b))

        # 通过 occurrence 单向链表获取所有出现 best_pair 的位置（左节点索引）
        # 从最新的 occurrence 开始遍历整个链表
        occ_idx = pair_latest_occ.get(best_pair, -1)
        if best_pair in pair_latest_occ:
            del pair_latest_occ[best_pair]   # 删除该 pair 的 occurrence 链头
        
        occ_nodes = []
        while occ_idx != -1:
            occ_nodes.append(pair_occ_list[occ_idx])
            occ_idx = pair_pre_occ_list[occ_idx]
            
        # 逆序处理，保证我们在同一个单词里从左向右处理合并（避免索引错乱）
        occ_nodes.reverse()

        dirty_pairs = set()   # 记录本次合并过程中发生变化的字节对，用于后续更新堆

        # 遍历每个出现位置，进行实际的合并操作
        for curr_idx in occ_nodes:
            # 检查节点有效性（可能已被之前的合并操作标记无效）
            if not node_valid[curr_idx]:
                continue
            nxt_idx = node_next[curr_idx]
            if nxt_idx == -1 or not node_valid[nxt_idx]:
                continue
            # 确认当前节点的 token 确实为 a，且下一个节点的 token 为 b（防御性检查）
            if node_val[curr_idx] != a or node_val[nxt_idx] != b:
                continue

            count = node_count[curr_idx]   # 该单词的频次
            prev_idx = node_prev[curr_idx]
            nxt_nxt_idx = node_next[nxt_idx]

            # 删除三个旧字节对（因为合并后它们不再存在）
            main_pair = (a, b)
            pairs[main_pair] -= count
            dirty_pairs.add(main_pair)

            if prev_idx != -1:
                prev_pair = (node_val[prev_idx], a)
                pairs[prev_pair] -= count
                dirty_pairs.add(prev_pair)

            if nxt_nxt_idx != -1:
                nxt_pair = (b, node_val[nxt_nxt_idx])
                pairs[nxt_pair] -= count
                dirty_pairs.add(nxt_pair)

            # 更新双向链表结构：将 curr_idx 节点值改为 ab，并跳过 nxt_idx
            node_val[curr_idx] = ab
            node_next[curr_idx] = nxt_nxt_idx
            if nxt_nxt_idx != -1:
                node_prev[nxt_nxt_idx] = curr_idx
            node_valid[nxt_idx] = False   # 标记 nxt_idx 节点为无效

            # 增加两个新字节对（如果存在）
            if prev_idx != -1:
                new_prev = (node_val[prev_idx], ab)
                pairs[new_prev] += count
                dirty_pairs.add(new_prev)
                
                # 将新 occurrence 追加到 occurrence 链表中
                new_occ_idx = len(pair_occ_list)
                pair_occ_list.append(prev_idx)
                pair_pre_occ_list.append(pair_latest_occ.get(new_prev, -1))
                pair_latest_occ[new_prev] = new_occ_idx

            if nxt_nxt_idx != -1:
                new_nxt = (ab, node_val[nxt_nxt_idx])
                pairs[new_nxt] += count
                dirty_pairs.add(new_nxt)
                
                # 追加新 occurrence
                new_occ_idx = len(pair_occ_list)
                pair_occ_list.append(curr_idx)
                pair_pre_occ_list.append(pair_latest_occ.get(new_nxt, -1))
                pair_latest_occ[new_nxt] = new_occ_idx

        # 一次性更新堆并清理 pairs 中的过期无效元素，减少迭代解包的开销
        for pair in dirty_pairs:
            count = pairs.get(pair, 0)
            if count <= 0:
                if pair in pairs:
                    del pairs[pair]
            else:
                # 频次 >0 的将新的 HeapNode 推入堆中
                # 旧的堆条目会被惰性忽略（因为频次不匹配）
                heapq.heappush(heap, HeapNode(count, pair))

    # ---------- 构建最终词表 ----------
    final_vocab = {i: _SINGLE_BYTE[i] for i in range(256)}   # 基础单字节词表
    next_id = 256

    # 为特殊 token 分配 ID（按给定顺序），注意这些 token 不参与 BPE 合并，直接添加
    if special_tokens:
        for st in special_tokens:
            final_vocab[next_id] = st.encode("utf-8")
            next_id += 1

    # 按合并顺序为每个合并结果分配 ID，每个合并结果就是 a+b 对应的字节串
    for a, b in merges:
        final_vocab[next_id] = a + b
        next_id += 1

    return final_vocab, merges


# ------------------------------------------------------------
class Tokenizer:
    """BPE 分词器，支持 encode / decode 以及特殊 token 处理。"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """
        初始化分词器。
        
        :param vocab: 词表，id -> 字节串
        :param merges: 合并规则列表，按训练顺序，每个元素为 (bytes, bytes)
        :param special_tokens: 特殊 token 列表（可选），用于编码时识别
        """
        self.id_to_bytes = vocab.copy()
        # 构建反向映射，用于编码时查找 token 的 id（字节串 -> id）
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        # 为 O(1) 的优先级查询构建 ranks 字典（越靠前的合并规则 rank 越小，优先级越高）
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        # 按长度降序存储特殊 token，使其在正则匹配时长的始终在前面，避免被截断
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.cache: dict[str, list[int]] = {}   # 单词缓存，加速重复单词的编码

        # 确保所有特殊 token 都在映射中（如果 vocab 中缺失则动态添加）
        # 这种情况可能发生在从文件加载词表时未包含特殊 token
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in self.bytes_to_id:
                new_id = max(self.id_to_bytes.keys()) + 1 if self.id_to_bytes else 0
                self.id_to_bytes[new_id] = b
                self.bytes_to_id[b] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        """
        从 pickle 文件加载之前训练好的 vocab 和 merges。
        
        :param vocab_filepath: 词汇表 pickle 文件路径
        :param merges_filepath: 合并规则 pickle 文件路径
        :param special_tokens: 可选的特殊 token 列表，若加载的 vocab 中未包含，则自动添加
        :return: Tokenizer 实例
        """
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为词元 id 列表。
        
        步骤：
        1. 按特殊 token 分割文本（保留分隔符）
        2. 对普通部分，使用 PAT 预分词得到单词
        3. 对每个单词，依次应用 merges 中的合并规则（贪心从左到右）
        4. 查找每个最终 token 的 id，并缓存结果
        """
        # 按特殊 token 分割文本（保留分隔符），使得特殊 token 独立成块
        if not self.special_tokens:
            parts = [text]
        else:
            escaped_st = [re.escape(st) for st in self.special_tokens]
            st_pattern = "|".join(escaped_st)
            parts = re.split(f"({st_pattern})", text)

        result = []
        for part in parts:
            if not part:
                continue
            # 如果当前部分是特殊 token，直接取其 id
            if self.special_tokens and part in self.special_tokens:
                result.append(self.bytes_to_id[part.encode("utf-8")])
                continue

            # 对普通文本部分，先用 PAT 预分词（使用 finditer 节省内存）
            for m in PAT.finditer(part):
                word_str = m.group(0)
                
                # 命中单词缓存，直接使用缓存结果
                if word_str in self.cache:
                    result.extend(self.cache[word_str])
                    continue

                # 将匹配的字符串转为字节，利用全局单字节缓存进行高效映射
                word_bytes = [_SINGLE_BYTE[b] for b in word_str.encode("utf-8")]

                # 利用 bpe_ranks 快速查找最高优先级的合法前缀对（基于 merges 中的排序）
                # 抛弃 O(V) 的全局 merges 遍历，使得时间复杂度大大降低
                while len(word_bytes) >= 2:
                    # 找出当前字符串中所有相邻的字节对
                    pairs = [(word_bytes[i], word_bytes[i+1]) for i in range(len(word_bytes)-1)]
                    
                    # 寻找 ranks 中排在最前面的 pair（即优先级最高的 merge）
                    # 对于不存在于 ranks 的 pair，如果没找到就会退出循环
                    best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                    if best_pair not in self.bpe_ranks:
                        break  # 如果没有任何 pair 在 bpe_ranks 中，说明无法再继续合并
                        
                    a, b = best_pair
                    new_word = []
                    i = 0
                    # 仅针对找到的这个最高优先级 best_pair 进行合并操作
                    while i < len(word_bytes):
                        if i < len(word_bytes)-1 and word_bytes[i] == a and word_bytes[i+1] == b:
                            new_word.append(a + b)
                            i += 2
                        else:
                            new_word.append(word_bytes[i])
                            i += 1
                    word_bytes = new_word   # 更新为合并后的序列，继续寻找下一个优先级对

                # 查找每个最终词元的 id 并加入结果
                token_ids = [self.bytes_to_id[token_bytes] for token_bytes in word_bytes]
                self.cache[word_str] = token_ids   # 缓存结果
                result.extend(token_ids)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        惰性编码：对可迭代的字符串序列进行缓冲拼接后分块编码，返回 ID 生成器。
        适合处理流式或大文件，不仅避免内存爆发，还通过自主控制分块大小
        降低对传入粒度（例如很短的文件行）的依赖，提高正则和 BPE 合并过程的吞吐率。
        
        :param iterable: 字符串的可迭代对象（例如文件行列表）
        :yield: 依次产生编码后的 id
        """
        buffer = []
        buffer_len = 0
        chunk_size_limit = 200 * 1024 * 1024  # 自定义缓冲块大小限制

        for text in iterable:
            buffer.append(text)
            buffer_len += len(text)
            
            # 当累积超过指定的块大小后进行一次性合并编码并 yield 结果
            if buffer_len >= chunk_size_limit:
                combined_text = "".join(buffer)
                yield from self.encode(combined_text)
                buffer.clear()
                buffer_len = 0

        # 处理并 yield 尾部剩余的一块文本
        if buffer:
            combined_text = "".join(buffer)
            yield from self.encode(combined_text)

    def decode(self, ids: list[int]) -> str:
        """
        将 id 列表解码回字符串。
        
        :param ids: 词元 id 列表
        :return: 解码后的 UTF-8 字符串，无效字节替换为 �
        """
        # 将所有 id 对应的字节串拼接成一个大字节串
        res_bytes = b"".join([self.id_to_bytes[i] for i in ids])
        # 按 UTF-8 解码，非法字节序列替换为 Unicode 替换字符 U+FFFD
        return res_bytes.decode("utf-8", errors="replace")

import json

# ------------------------------------------------------------------
# 辅助函数：加载 JSON 格式的 vocab 和 merges
def load_vocab_json(vocab_path: str) -> dict[int, bytes]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    return {int(k): v.encode("latin-1") for k, v in vocab_dict.items()}

def load_merges_json(merges_path: str) -> list[tuple[bytes, bytes]]:
    with open(merges_path, "r", encoding="utf-8") as f:
        merges_list = json.load(f)
    return [(a.encode("latin-1"), b.encode("latin-1")) for a, b in merges_list]

# ------------------------------------------------------------------
# 多进程编码的核心函数（模块级 worker 函数）
_tokenizer = None  # 模块级全局变量

def _init_worker(vocab_path: str, merges_path: str, st: list[str]):
    """子进程初始化：加载 tokenizer"""
    global _tokenizer
    vocab = load_vocab_json(vocab_path)
    merges = load_merges_json(merges_path)
    _tokenizer = Tokenizer(vocab, merges, special_tokens=st)
def _encode_file_chunk(args):
    """子进程工作函数：编码文件的一个块（由边界 start/end 指定）"""
    input_path, start, end, vocab_path, merges_path, st = args
    global _tokenizer
    # 如果 _tokenizer 尚未初始化（例如 initializer 未执行），则初始化
    if _tokenizer is None:
        vocab = load_vocab_json(vocab_path)
        merges = load_merges_json(merges_path)
        _tokenizer = Tokenizer(vocab, merges, special_tokens=st)
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode('utf-8', errors='ignore')
    return _tokenizer.encode(text)
# def _encode_chunk(chunk: str) -> list[int]:
#     """子进程工作函数：编码一个文本块"""
#     global _tokenizer
#     if _tokenizer is None:
#         raise RuntimeError("Tokenizer not initialized in worker")
#     return _tokenizer.encode(chunk)

# def parallel_encode_iterable(
#     vocab_json_path: str,
#     merges_json_path: str,
#     special_tokens: list[str],
#     text_iterable: Iterable[str],
#     num_processes: int = 5,
#     use_ordered: bool = True
# ) -> Iterator[int]:
#     """
#     多进程并行编码文本块迭代器，返回 token ID 生成器（保持输入顺序）。

#     参数：
#         vocab_json_path: 词汇表 JSON 文件路径
#         merges_json_path: merges JSON 文件路径
#         special_tokens: 特殊 token 列表（必须包含 "<|endoftext|>" 等）
#         text_iterable: 文本块的可迭代对象（每个元素是一个完整的字符串块，建议大小 10~50 MB）
#         num_processes: 并行进程数（建议 ≤ CPU 核数，默认 5）
#         use_ordered: 是否保持顺序（True 则使用 imap，False 则使用 imap_unordered 可能更快但乱序）

#     产出：
#         int - 依次产生的 token ID
#     """
#     with multiprocessing.Pool(
#         processes=num_processes,
#         initializer=_init_worker,
#         initargs=(vocab_json_path, merges_json_path, special_tokens)
#     ) as pool:
#         if use_ordered:
#             for ids in pool.imap(_encode_chunk, text_iterable):
#                 yield ids
#         else:
#             for ids in pool.imap_unordered(_encode_chunk, text_iterable):
#                 yield ids

def parallel_encode_file(
    input_path: str,
    vocab_json_path: str,
    merges_json_path: str,
    output_bin_path: str,
    special_tokens: list[str],
    num_processes: int = 6,
    chunk_target_size: int = 100 * 1024 * 1024
) -> int:
    """多进程直接按文件块编码（不经过主进程生成器），返回总 token 数"""
    # 确保有 split_token
    split_token = "<|endoftext|>"
    if split_token not in special_tokens:
        raise ValueError("special_tokens must contain '<|endoftext|>'")
    
    file_size = os.path.getsize(input_path)
    desired_chunks = max(num_processes, file_size // chunk_target_size)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, split_token.encode())
    
    # 准备任务参数：每个块 (input_path, start, end, vocab_path, merges_path, special_tokens)
    args_list = [
        (input_path, boundaries[i], boundaries[i+1], vocab_json_path, merges_json_path, special_tokens)
        for i in range(len(boundaries)-1)
    ]
    
    
    with multiprocessing.Pool(processes=num_processes, initializer=_init_worker,
                              initargs=(vocab_json_path, merges_json_path, special_tokens)) as pool:
        total = 0
        with open(output_bin_path, "wb") as outf:
            for token_list in pool.imap(_encode_file_chunk, args_list):
                if token_list:
                    arr = np.array(token_list, dtype=np.uint16)
                    outf.write(arr.tobytes())
                    total += len(token_list)
    return total