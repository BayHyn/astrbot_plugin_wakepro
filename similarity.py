import math
import jieba
from collections import defaultdict, deque
import re


class Similarity:
    """动态话题关联性检测器"""

    # 基础停用词表
    STOP = {
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "或",
        "这",
        "那",
        "我",
        "你",
        "他",
        "她",
        "它",
    }

    # 话题缓存系统
    TOPIC_CACHE = deque(maxlen=20)  # 保存最近20个话题关键词
    TOPIC_WEIGHTS = defaultdict(float)  # 动态权重字典
    DECAY_FACTOR = 0.95  # 权重衰减因子

    @classmethod
    def _update_topic_cache(cls, words):
        """更新话题缓存和权重"""
        # 更新缓存
        for word in words:
            # 只缓存非停用词且长度>1的实词
            if (
                word not in cls.STOP
                and len(word) > 1
                and re.match(r"^[\u4e00-\u9fa5]+$", word)
            ):
                cls.TOPIC_CACHE.append(word)

        # 计算最近话题频率
        freq_counter = defaultdict(int)
        for word in cls.TOPIC_CACHE:
            freq_counter[word] += 1

        # 更新权重（新词更高权重）
        for word, count in freq_counter.items():
            # 新话题获得更高权重，旧话题逐渐衰减
            decayed_weight = cls.TOPIC_WEIGHTS.get(word, 0) * cls.DECAY_FACTOR
            current_weight = count * (1.0 + math.log(len(word)))  # 长词权重更高
            cls.TOPIC_WEIGHTS[word] = max(decayed_weight, current_weight)

    @classmethod
    def _extract_keywords(cls, s: str) -> list:
        """提取关键词并更新话题缓存"""
        # 文本清洗
        s = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", s)  # 保留中文字符和数字
        words = [w for w in jieba.lcut(s) if w.strip() and w not in cls.STOP]

        # 合并连续数字和专有名词
        merged = []
        for word in words:
            if merged and (
                (word.isdigit() and merged[-1][-1].isdigit())
                or (len(merged[-1]) == 1 and len(word) == 1)
            ):
                merged[-1] += word
            else:
                merged.append(word)

        # 更新话题缓存
        cls._update_topic_cache(merged)
        return merged

    @classmethod
    def _tokens(cls, s: str) -> dict[str, float]:
        """生成带权重的词向量"""
        words = cls._extract_keywords(s)
        tf = defaultdict(float)

        # 添加一元词（应用动态权重）
        for word in words:
            base_weight = 1.0
            # 应用话题权重（1.0 + 缓存权重）
            weight = base_weight + cls.TOPIC_WEIGHTS.get(word, 0)
            tf[word] += weight

        # 添加二元词组（长距离关联）
        for i in range(len(words) - 1):
            bigram = words[i] + words[i + 1]
            # 二元组权重更高（1.5倍）
            tf[bigram] += 1.5

        # 归一化处理
        total = max(sum(tf.values()), 1)
        return {w: count / total for w, count in tf.items()}

    @classmethod
    def cosine(cls, a: str, b: str) -> float:
        """计算话题关联性得分（0-1范围）"""
        v1, v2 = cls._tokens(a), cls._tokens(b)
        all_w = set(v1) | set(v2)

        # 计算增强点积（共同词权重加倍）
        dot_product = 0
        for w in all_w:
            val1 = v1.get(w, 0)
            val2 = v2.get(w, 0)
            # 关键优化：共同词获得额外权重
            if val1 > 0 and val2 > 0:
                dot_product += val1 * val2 * (2.0 + cls.TOPIC_WEIGHTS.get(w, 0))
            else:
                dot_product += val1 * val2

        # 计算模长
        norm1 = math.sqrt(sum(val**2 for val in v1.values()))
        norm2 = math.sqrt(sum(val**2 for val in v2.values()))

        # 防止除零错误
        denominator = norm1 * norm2 + 1e-8

        # 计算原始相似度
        raw_score = dot_product / denominator

        # 应用Sigmoid函数使结果更平滑
        return 1 / (1 + math.exp(-8 * (raw_score - 0.6)))

    @classmethod
    def get_current_topics(cls, top_n=5):
        """获取当前最重要的top_n个话题"""
        return sorted(cls.TOPIC_WEIGHTS.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
