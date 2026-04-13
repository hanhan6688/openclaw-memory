"""
OpenClaw Memory System - 高级检索模块
================================

功能：
1. 时间衰减: 近期记忆权重更高
2. 混合检索: Weaviate 原生 hybrid (向量 + BM25)
3. 查询意图识别: 模糊匹配 vs 精确回忆

使用 Weaviate 原生混合检索:
- collection.query.hybrid(query, alpha=0.5)
- alpha=0: 纯 BM25
- alpha=1: 纯向量
- alpha=0.5: 平衡混合
"""

import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional


class TimeDecay:
    """
    时间衰减计算器
    
    近期记忆权重更高，随时间逐渐衰减
    
    公式: weight = base_weight × decay_factor^(days_ago)
    
    示例:
        今天: 1.2x
        昨天: 0.9x
        7天前: 0.5x
        30天前: 0.2x
    """
    
    def __init__(self, 
                 recent_boost: float = 1.2,  # 近期加成
                 decay_rate: float = 0.9,     # 衰减率
                 half_life_days: float = 7.0):  # 半衰期
        self.recent_boost = recent_boost
        self.decay_rate = decay_rate
        self.half_life_days = half_life_days

    def _normalize_datetime(self, value: datetime) -> datetime:
        """将时间统一为 UTC aware datetime。"""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _parse_timestamp(self, timestamp) -> Optional[datetime]:
        """解析字符串或 datetime 时间戳。"""
        if isinstance(timestamp, datetime):
            return self._normalize_datetime(timestamp)

        if not isinstance(timestamp, str) or not timestamp.strip():
            return None

        normalized = timestamp.strip().replace("Z", "+00:00")

        try:
            return self._normalize_datetime(datetime.fromisoformat(normalized))
        except ValueError:
            pass

        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return self._normalize_datetime(datetime.strptime(normalized, fmt))
            except ValueError:
                continue

        return None
    
    def calculate_weight(self, timestamp: str, base_importance: float = 0.5, reference_time: datetime = None) -> float:
        """
        计算时间衰减后的权重
        
        Args:
            timestamp: ISO 格式时间戳
            base_importance: 基础重要性 (0-1)
            reference_time: 参考时间，如果为None则使用当前时间
        
        Returns:
            衰减后的权重
        """
        try:
            memory_time = self._parse_timestamp(timestamp)
            if memory_time is None:
                return base_importance
            
            # 计算天数差
            now = reference_time if reference_time is not None else datetime.now(timezone.utc)
            now = self._normalize_datetime(now)
            days_ago = (now - memory_time).total_seconds() / 86400
            days_ago = max(days_ago, 0.0)
            
            # 近期加成（24小时内）
            if days_ago < 1:
                time_weight = self.recent_boost
            else:
                # 指数衰减
                time_weight = self.decay_rate ** (days_ago / self.half_life_days)
            
            # 综合权重 = 基础重要性 × 时间权重
            final_weight = base_importance * time_weight
            
            return min(final_weight, 1.0)
            
        except Exception:
            return base_importance
    
    def get_boost_factor(self, days_ago: float, reference_time: datetime = None) -> float:
        """获取指定天数前的加成因子"""
        # Note: This method is kept for backward compatibility but reference_time is ignored
        # as days_ago is already calculated
        if days_ago < 1:
            return self.recent_boost
        return self.decay_rate ** (days_ago / self.half_life_days)


class QueryIntentRecognizer:
    """
    查询意图识别器
    
    识别用户是想"模糊匹配"还是"精确回忆"
    """
    
    # 模糊匹配关键词
    FUZZY_KEYWORDS = [
        "好像", "大概", "可能", "似乎", "类似", "差不多",
        "记得", "印象", "模糊", "应该", "什么来着"
    ]
    FUZZY_KEYWORDS_EN = [
        "roughly", "maybe", "perhaps", "seems", "vaguely",
        "i remember", "something like", "what was it again",
        "kind of", "sort of"
    ]
    
    # 精确回忆关键词
    EXACT_KEYWORDS = [
        "具体", "精确", "原话", "确切", "一字不差",
        "原文", "完整", "详细", "准确"
    ]
    EXACT_KEYWORDS_EN = [
        "exact", "precise", "verbatim", "word for word",
        "original wording", "original text", "full text",
        "complete", "accurate"
    ]
    
    # 时间相关关键词
    TIME_KEYWORDS = [
        "今天", "昨天", "前天", "最近", "刚才",
        "上周", "上个月", "去年", "之前"
    ]
    TIME_KEYWORDS_EN = [
        "today", "yesterday", "the day before yesterday",
        "recently", "these days", "just now",
        "last week", "last month", "last year", "earlier"
    ]
    RANGE_KEYWORDS = ["最近", "这几天", "上周", "上个月", "去年", "刚才", "刚刚", "这周", "这个月", "今年"]
    RANGE_KEYWORDS_EN = [
        "recently", "these days", "just now", "right now",
        "last week", "last month", "last year",
        "this week", "this month", "this year",
    ]

    RELATIVE_TIME_PATTERN = re.compile(r"(\d+)\s*(分钟|小时|天|周|星期|个月|月|年)前|半(小时|天|年)前")
    RELATIVE_TIME_PATTERN_EN = re.compile(
        r"\b(\d+)\s+(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago\b",
        re.IGNORECASE,
    )
    HALF_HOUR_PATTERN_EN = re.compile(r"\bhalf(?:\s+an)?\s+hour\s+ago\b|\bhalf-hour\s+ago\b", re.IGNORECASE)
    HALF_DAY_PATTERN_EN = re.compile(r"\bhalf(?:\s+a)?\s+day\s+ago\b|\bhalf-day\s+ago\b", re.IGNORECASE)
    HALF_YEAR_PATTERN_EN = re.compile(
        r"\bhalf(?:\s+a)?\s+year\s+ago\b|\bhalf-year\s+ago\b",
        re.IGNORECASE,
    )
    TIME_CLEANUP_PATTERNS = [
        re.compile(r"\b(?:from|in|on|at|during|around|about)\s+(?=(today|yesterday|the day before yesterday|recently|these days|just now|right now|last week|last month|last year|this week|this month|this year|\d+\s+(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago|half(?:\s+(?:an|a))?\s+(?:hour|day|year)\s+ago))", re.IGNORECASE),
    ]

    def _contains_any(self, query: str, keywords: List[str], *, lowercase: bool = False) -> bool:
        haystack = query.lower() if lowercase else query
        return any(keyword in haystack for keyword in keywords)

    def _collect_time_keywords(self, query: str, query_lower: str) -> List[str]:
        hits = [kw for kw in self.TIME_KEYWORDS if kw in query]
        hits.extend(kw for kw in self.TIME_KEYWORDS_EN if kw in query_lower)
        return hits

    def _start_of_hour(self, dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    def _start_of_day(self, dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def _end_of_day(self, dt: datetime) -> datetime:
        return dt.replace(hour=23, minute=59, second=59, microsecond=0)

    def _start_of_week(self, dt: datetime) -> datetime:
        return self._start_of_day(dt - timedelta(days=dt.weekday()))

    def _start_of_month(self, dt: datetime) -> datetime:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _start_of_year(self, dt: datetime) -> datetime:
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def _normalize_cleaned_query(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([?.!,])", r"\1", text)
        text = re.sub(r"^[的了在关于有关]+\s*", "", text)
        text = re.sub(r"\s*(的|了|呢|呀|吧)+$", "", text)
        text = re.sub(r"\b(?:from|in|on|at|during|around|about)\b\s*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^[,，.。!?！？:：;\- ]+|[,，.。!?！？:：;\- ]+$", "", text)
        return text.strip()

    def _strip_time_expressions(self, query: str, query_lower: str) -> str:
        cleaned = query
        cleanup_phrases = (
            self.TIME_KEYWORDS
            + self.TIME_KEYWORDS_EN
            + self.RANGE_KEYWORDS
            + self.RANGE_KEYWORDS_EN
            + ["刚刚", "这周", "这个月", "今年"]
        )
        for phrase in sorted(set(cleanup_phrases), key=len, reverse=True):
            cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)

        cleaned = self.RELATIVE_TIME_PATTERN.sub(" ", cleaned)
        cleaned = self.RELATIVE_TIME_PATTERN_EN.sub(" ", cleaned)
        cleaned = self.HALF_HOUR_PATTERN_EN.sub(" ", cleaned)
        cleaned = self.HALF_DAY_PATTERN_EN.sub(" ", cleaned)
        cleaned = self.HALF_YEAR_PATTERN_EN.sub(" ", cleaned)

        for pattern in self.TIME_CLEANUP_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        return self._normalize_cleaned_query(cleaned)

    def _extract_relative_offset(self, query: str, query_lower: str) -> Optional[Tuple[timedelta, str]]:
        relative_match = re.search(r"(\d+)\s*(分钟|小时|天|周|星期|个月|月|年)前", query)
        if relative_match:
            amount = int(relative_match.group(1))
            unit = relative_match.group(2)
            mapping = {
                "分钟": (timedelta(minutes=amount), "minute"),
                "小时": (timedelta(hours=amount), "hour"),
                "天": (timedelta(days=amount), "day"),
                "周": (timedelta(weeks=amount), "week"),
                "星期": (timedelta(weeks=amount), "week"),
                "个月": (timedelta(days=amount * 30), "month"),
                "月": (timedelta(days=amount * 30), "month"),
                "年": (timedelta(days=amount * 365), "year"),
            }
            return mapping[unit]

        if "半小时前" in query:
            return timedelta(minutes=30), "minute"
        if "半天前" in query:
            return timedelta(hours=12), "hour"
        if "半年前" in query:
            return timedelta(days=180), "month"

        english_match = self.RELATIVE_TIME_PATTERN_EN.search(query_lower)
        if english_match:
            amount = int(english_match.group(1))
            unit = english_match.group(2).lower()
            if unit.startswith("minute"):
                return timedelta(minutes=amount), "minute"
            if unit.startswith("hour"):
                return timedelta(hours=amount), "hour"
            if unit.startswith("day"):
                return timedelta(days=amount), "day"
            if unit.startswith("week"):
                return timedelta(weeks=amount), "week"
            if unit.startswith("month"):
                return timedelta(days=amount * 30), "month"
            if unit.startswith("year"):
                return timedelta(days=amount * 365), "year"

        if self.HALF_HOUR_PATTERN_EN.search(query_lower):
            return timedelta(minutes=30), "minute"
        if self.HALF_DAY_PATTERN_EN.search(query_lower):
            return timedelta(hours=12), "hour"
        if self.HALF_YEAR_PATTERN_EN.search(query_lower):
            return timedelta(days=180), "month"

        return None

    def extract_time_anchor(self, query: str) -> Dict:
        """提取结构化时间锚点信息。"""
        now = datetime.now(timezone.utc)
        query_lower = query.lower()
        cleaned_query = self._strip_time_expressions(query, query_lower)

        anchor = {
            "reference_time": None,
            "time_range": None,
            "anchor_type": None,
            "anchor_granularity": None,
            "cleaned_query": cleaned_query,
        }

        if "今天" in query or "today" in query_lower:
            anchor.update(
                reference_time=self._start_of_day(now),
                time_range=(self._start_of_day(now), now),
                anchor_type="range",
                anchor_granularity="day",
            )
            return anchor

        if "昨天" in query or "yesterday" in query_lower:
            target = now - timedelta(days=1)
            anchor.update(
                reference_time=self._start_of_day(target),
                time_range=(self._start_of_day(target), self._end_of_day(target)),
                anchor_type="point",
                anchor_granularity="day",
            )
            return anchor

        if "前天" in query or "the day before yesterday" in query_lower:
            target = now - timedelta(days=2)
            anchor.update(
                reference_time=self._start_of_day(target),
                time_range=(self._start_of_day(target), self._end_of_day(target)),
                anchor_type="point",
                anchor_granularity="day",
            )
            return anchor

        relative_offset = self._extract_relative_offset(query, query_lower)
        if relative_offset is not None:
            offset, granularity = relative_offset
            target = now - offset
            if granularity == "minute":
                start = target.replace(second=0, microsecond=0)
                end = start + timedelta(minutes=1)
            elif granularity == "hour":
                start = self._start_of_hour(target)
                end = start + timedelta(hours=1)
            elif granularity == "day":
                start = self._start_of_day(target)
                end = self._end_of_day(target)
            elif granularity == "week":
                start = self._start_of_day(target)
                end = start + timedelta(days=1)
            elif granularity == "month":
                start = self._start_of_day(target)
                end = self._end_of_day(target)
            else:
                start = self._start_of_day(target)
                end = self._end_of_day(target)

            anchor.update(
                reference_time=start,
                time_range=(start, end),
                anchor_type="point",
                anchor_granularity=granularity,
            )
            return anchor

        if "刚才" in query or "刚刚" in query or "just now" in query_lower or "right now" in query_lower:
            start = now - timedelta(hours=1)
            anchor.update(
                reference_time=start,
                time_range=(start, now),
                anchor_type="range",
                anchor_granularity="hour",
            )
            return anchor

        if "最近" in query or "这几天" in query or "recently" in query_lower or "these days" in query_lower:
            start = now - timedelta(days=7)
            anchor.update(
                reference_time=self._start_of_day(start),
                time_range=(self._start_of_day(start), now),
                anchor_type="range",
                anchor_granularity="day",
            )
            return anchor

        if "上周" in query or "last week" in query_lower:
            start = self._start_of_week(now) - timedelta(days=7)
            end = self._start_of_week(now)
            anchor.update(
                reference_time=start,
                time_range=(start, end),
                anchor_type="range",
                anchor_granularity="week",
            )
            return anchor

        if "这周" in query or "this week" in query_lower:
            start = self._start_of_week(now)
            anchor.update(
                reference_time=start,
                time_range=(start, now),
                anchor_type="range",
                anchor_granularity="week",
            )
            return anchor

        if "上个月" in query or "last month" in query_lower:
            current_month_start = self._start_of_month(now)
            last_month_end = current_month_start
            last_month_start = self._start_of_month(current_month_start - timedelta(days=1))
            anchor.update(
                reference_time=last_month_start,
                time_range=(last_month_start, last_month_end),
                anchor_type="range",
                anchor_granularity="month",
            )
            return anchor

        if "这个月" in query or "this month" in query_lower:
            start = self._start_of_month(now)
            anchor.update(
                reference_time=start,
                time_range=(start, now),
                anchor_type="range",
                anchor_granularity="month",
            )
            return anchor

        if "去年" in query or "last year" in query_lower:
            current_year_start = self._start_of_year(now)
            last_year_start = current_year_start.replace(year=current_year_start.year - 1)
            anchor.update(
                reference_time=last_year_start,
                time_range=(last_year_start, current_year_start),
                anchor_type="range",
                anchor_granularity="year",
            )
            return anchor

        if "今年" in query or "this year" in query_lower:
            start = self._start_of_year(now)
            anchor.update(
                reference_time=start,
                time_range=(start, now),
                anchor_type="range",
                anchor_granularity="year",
            )
            return anchor

        return anchor

    def _contains_time_expression(self, query: str) -> bool:
        """判断查询是否包含显式时间表达。"""
        anchor = self.extract_time_anchor(query)
        return anchor["reference_time"] is not None or anchor["time_range"] is not None
    
    def recognize(self, query: str) -> Dict:
        """
        识别查询意图
        
        Returns:
            {
                "intent": "fuzzy" | "exact" | "time" | "general",
                "confidence": 0.0-1.0,
                "suggested_mode": "vector" | "bm25" | "hybrid" | "time",
                "time_hint": "今天" | "最近" | None,
                "reference_time": datetime | None,
                "reason": "原因说明"
            }
        """
        query_lower = query.lower()

        # 检查引号（精确引用）
        if any(marker in query for marker in ['"', "“", "”", "「", "」"]):
            return {
                "intent": "exact",
                "confidence": 0.9,
                "suggested_mode": "bm25",
                "time_hint": None,
                "reference_time": None,
                "time_range": None,
                "anchor_type": None,
                "anchor_granularity": None,
                "cleaned_query": query.strip(),
                "reason": "查询包含引号，可能是精确引用"
            }
        
        # 检查精确关键词
        if self._contains_any(query, self.EXACT_KEYWORDS) or self._contains_any(
            query_lower, self.EXACT_KEYWORDS_EN
        ):
            return {
                "intent": "exact",
                "confidence": 0.8,
                "suggested_mode": "bm25",
                "time_hint": None,
                "reference_time": None,
                "time_range": None,
                "anchor_type": None,
                "anchor_granularity": None,
                "cleaned_query": query.strip(),
                "reason": "包含精确关键词"
            }
        
        # 检查时间关键词
        time_keywords = self._collect_time_keywords(query, query_lower)
        time_anchor = self.extract_time_anchor(query)
        if self._contains_time_expression(query):
            return {
                "intent": "time",
                "confidence": 0.85,
                "suggested_mode": "time",
                "time_hint": time_keywords[0] if time_keywords else None,
                "reference_time": time_anchor["reference_time"],
                "time_range": time_anchor["time_range"],
                "anchor_type": time_anchor["anchor_type"],
                "anchor_granularity": time_anchor["anchor_granularity"],
                "cleaned_query": time_anchor["cleaned_query"],
                "reason": (
                    f"包含时间关键词: {time_keywords}"
                    if time_keywords
                    else "包含相对时间表达"
                )
            }
        
        # 检查模糊关键词
        if self._contains_any(query, self.FUZZY_KEYWORDS) or self._contains_any(
            query_lower, self.FUZZY_KEYWORDS_EN
        ):
            return {
                "intent": "fuzzy",
                "confidence": 0.7,
                "suggested_mode": "vector",
                "time_hint": None,
                "reference_time": None,
                "time_range": None,
                "anchor_type": None,
                "anchor_granularity": None,
                "cleaned_query": query.strip(),
                "reason": "包含模糊关键词"
            }
        
        # 默认：混合模式
        return {
            "intent": "general",
            "confidence": 0.5,
            "suggested_mode": "hybrid",
            "time_hint": None,
            "reference_time": None,
            "time_range": None,
            "anchor_type": None,
            "anchor_granularity": None,
            "cleaned_query": query.strip(),
            "reason": "通用查询，建议混合检索"
        }
    
    def extract_reference_time(self, query: str) -> Optional[datetime]:
        """从查询中提取参考时间，用于时间衰减计算"""
        return self.extract_time_anchor(query).get("reference_time")
    
    def extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """从查询中提取时间范围"""
        return self.extract_time_anchor(query).get("time_range")


class HybridRetriever:
    """
    混合检索器
    
    使用 Weaviate 原生混合检索 (向量 + BM25)
    """

    def __init__(self, memory_store):
        self.memory_store = memory_store
        self.time_decay = TimeDecay()
        self.intent_recognizer = QueryIntentRecognizer()
        
        # alpha 参数: 0=纯BM25, 1=纯向量, 0.5=平衡
        self.default_alpha = 0.5

    def search(self, query: str, limit: int = 10, mode: str = "auto", time_reference: datetime = None) -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询文本
            limit: 返回数量
            mode: 检索模式
                - "auto": 自动识别意图
                - "vector": 纯向量检索 (alpha=1)
                - "bm25": 纯 BM25 检索 (alpha=0)
                - "hybrid": 混合检索 (alpha=0.5)
                - "time": 时间优先检索
            time_reference: 参考时间，用于计算时间衰减，如果为None则使用当前时间

        Returns:
            检索结果列表
        """
        # 意图识别
        intent_result = self.intent_recognizer.recognize(query)
        effective_query = intent_result.get("cleaned_query", query).strip()
        
        # 如果意图识别提供了参考时间，则使用它（除非显式提供了time_reference参数）
        if time_reference is None and intent_result.get("reference_time") is not None:
            time_reference = intent_result["reference_time"]

        # 确定检索模式
        if mode == "auto":
            mode = intent_result["suggested_mode"]

        # 根据模式选择检索策略
        if mode == "vector":
            return self._vector_search(effective_query or query, limit, time_reference)
        elif mode == "bm25":
            return self._bm25_search(effective_query or query, limit, time_reference)
        elif mode == "time":
            time_range = intent_result.get("time_range") or self.intent_recognizer.extract_time_range(query)
            time_query = effective_query
            if time_range:
                return self._time_search(time_query, time_range, limit, time_reference)
            else:
                return self._hybrid_search(effective_query or query, limit, time_reference)
        else:  # hybrid
            return self._hybrid_search(effective_query or query, limit, time_reference)

    def _vector_search(self, query: str, limit: int, time_reference: datetime = None) -> List[Dict]:
        """纯向量检索 (alpha=1)"""
        return self._weaviate_hybrid_search(query, alpha=1.0, limit=limit, time_reference=time_reference)

    def _bm25_search(self, query: str, limit: int, time_reference: datetime = None) -> List[Dict]:
        """纯 BM25 检索 (alpha=0)"""
        return self._weaviate_hybrid_search(query, alpha=0.0, limit=limit, time_reference=time_reference)

    def _hybrid_search(self, query: str, limit: int, time_reference: datetime = None) -> List[Dict]:
        """混合检索 (alpha=0.5) - 使用 Weaviate 原生"""
        return self._weaviate_hybrid_search(query, alpha=self.default_alpha, limit=limit, time_reference=time_reference)
    
    def _weaviate_hybrid_search(
        self,
        query: str,
        alpha: float = 0.5,
        limit: int = 10,
        time_reference: datetime = None,
    ) -> List[Dict]:
        """
        使用 Weaviate 原生混合检索
        
        Args:
            query: 查询文本
            alpha: 混合权重 (0=BM25, 1=向量, 0.5=平衡)
            limit: 返回数量
        """
        # 使用 Weaviate 原生混合检索
        results = self.memory_store.client.hybrid_search(query, alpha=alpha, limit=limit)
        
        # 应用时间衰减
        for r in results:
            r["time_weight"] = self.time_decay.calculate_weight(
                r.get("timestamp", ""),
                r.get("importance", 0.5),
                reference_time=time_reference
            )
            # 设置检索模式标记
            if alpha == 1.0:
                r["search_mode"] = "vector"
            elif alpha == 0.0:
                r["search_mode"] = "bm25"
            else:
                r["search_mode"] = "hybrid"
                r["alpha"] = alpha
        
        return results

    def _time_search(self, query: str, time_range: Tuple[datetime, datetime], limit: int, time_reference: datetime = None) -> List[Dict]:
        """时间优先检索"""
        start_time, end_time = time_range

        # 格式化时间
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        # 获取时间范围内的记忆
        memories = self.memory_store.get_by_time_range(start_str, end_str, limit * 2)

        # 如果有查询词，使用混合检索过滤
        if query:
            # 使用混合检索
            hybrid_results = self._hybrid_search(query, limit * 2, time_reference)
            hybrid_ids = {r.get("id") for r in hybrid_results}
            memories = [m for m in memories if m.get("id") in hybrid_ids]

        # 应用时间衰减
        for m in memories:
            m["time_weight"] = self.time_decay.calculate_weight(
                m.get("timestamp", ""),
                m.get("importance", 0.5),
                reference_time=time_reference
            )

        return memories[:limit]


# ==================== 便捷函数 ====================

def create_hybrid_retriever(memory_store) -> HybridRetriever:
    """创建混合检索器"""
    return HybridRetriever(memory_store)


# 示例用法
if __name__ == "__main__":
    # 测试时间衰减
    decay = TimeDecay()
    print("时间衰减测试:")
    print(f"  今天: {decay.get_boost_factor(0):.2f}x")
    print(f"  昨天: {decay.get_boost_factor(1):.2f}x")
    print(f"  7天前: {decay.get_boost_factor(7):.2f}x")
    print(f"  30天前: {decay.get_boost_factor(30):.2f}x")
    
    # 测试意图识别
    recognizer = QueryIntentRecognizer()
    test_queries = [
        "好像说过什么来着",
        "具体说了什么",
        "昨天的内容",
        "产品需求文档",
        "3天前发生了什么",
        "半年前我们讨论过的项目",
        "上周五的会议内容",
        "2个月前决定的方案"
    ]
    
    print("\n意图识别测试:")
    for q in test_queries:
        result = recognizer.recognize(q)
        print(f"  '{q}' -> {result['intent']} ({result['suggested_mode']})")
        if result.get('reference_time'):
            print(f"    参考时间: {result['reference_time'].strftime('%Y-%m-%d')}")
