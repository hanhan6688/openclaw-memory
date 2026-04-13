from datetime import datetime, timedelta, timezone
from math import isclose

import openclaw_memory
from openclaw_memory.core.hybrid_retrieval import HybridRetriever, QueryIntentRecognizer, TimeDecay


class _FakeClient:
    def __init__(self, results):
        self._results = results
        self.last_query = None
        self.last_alpha = None
        self.last_limit = None

    def hybrid_search(self, query, alpha=0.5, limit=10):
        self.last_query = query
        self.last_alpha = alpha
        self.last_limit = limit
        return [dict(item) for item in self._results[:limit]]


class _FakeMemoryStore:
    def __init__(self, results=None, time_range_results=None):
        self.client = _FakeClient(results or [])
        self._time_range_results = time_range_results or []

    def get_by_time_range(self, start, end, limit):
        return [dict(item) for item in self._time_range_results[:limit]]


def test_package_import_is_lazy():
    assert openclaw_memory.__version__ == "1.0.0"
    assert "MemoryStore" in dir(openclaw_memory)
    assert "MemoryStore" not in openclaw_memory.__dict__


def test_time_decay_parses_naive_timestamp_as_utc():
    decay = TimeDecay()
    reference_time = datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)

    weight = decay.calculate_weight(
        "2026-04-13T12:00:00",
        base_importance=0.5,
        reference_time=reference_time,
    )

    assert isclose(weight, 0.6, rel_tol=1e-9)


def test_time_decay_respects_reference_time():
    decay = TimeDecay()
    timestamp = datetime(2026, 4, 12, 12, 0, 0, tzinfo=timezone.utc)

    weight_at_event_time = decay.calculate_weight(timestamp, 0.5, reference_time=timestamp)
    weight_one_day_later = decay.calculate_weight(timestamp, 0.5, reference_time=timestamp + timedelta(days=1))

    assert weight_at_event_time > weight_one_day_later


def test_query_intent_recognizer_extracts_relative_time():
    recognizer = QueryIntentRecognizer()
    result = recognizer.recognize("2周前的讨论结论是什么")

    assert result["intent"] == "time"
    assert result["suggested_mode"] == "time"
    expected_date = datetime.now(timezone.utc) - timedelta(weeks=2)
    expected_date = expected_date.replace(hour=0, minute=0, second=0, microsecond=0)
    assert result["reference_time"].date() == expected_date.date()


def test_query_intent_recognizer_extracts_english_relative_time():
    recognizer = QueryIntentRecognizer()
    result = recognizer.recognize("What did we decide 3 days ago?")

    assert result["intent"] == "time"
    assert result["suggested_mode"] == "time"
    expected_date = datetime.now(timezone.utc) - timedelta(days=3)
    expected_date = expected_date.replace(hour=0, minute=0, second=0, microsecond=0)
    assert result["reference_time"].date() == expected_date.date()
    assert result["cleaned_query"] == "What did we decide"


def test_query_intent_recognizer_supports_english_exact_and_fuzzy_queries():
    recognizer = QueryIntentRecognizer()

    exact_result = recognizer.recognize("Give me the exact wording from last week")
    fuzzy_result = recognizer.recognize("I vaguely remember something like that")

    assert exact_result["intent"] == "exact"
    assert fuzzy_result["intent"] == "fuzzy"


def test_query_intent_recognizer_extracts_structured_month_anchor():
    recognizer = QueryIntentRecognizer()
    result = recognizer.recognize("The roadmap from last month")

    assert result["intent"] == "time"
    assert result["anchor_type"] == "range"
    assert result["anchor_granularity"] == "month"
    assert result["cleaned_query"] == "The roadmap"

    now = datetime.now(timezone.utc)
    current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_end = current_month_start
    previous_month_last_day = current_month_start - timedelta(days=1)
    last_month_start = previous_month_last_day.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    actual_start, actual_end = result["time_range"]
    assert actual_start == last_month_start
    assert actual_end == last_month_end


def test_query_intent_recognizer_supports_hour_level_anchor():
    recognizer = QueryIntentRecognizer()
    result = recognizer.recognize("What changed 2 hours ago?")

    assert result["intent"] == "time"
    assert result["anchor_granularity"] == "hour"
    assert result["time_range"] is not None
    assert result["cleaned_query"] == "What changed"


def test_hybrid_retriever_applies_reference_time_weight():
    reference_time = datetime(2026, 4, 10, 9, 0, 0, tzinfo=timezone.utc)
    store = _FakeMemoryStore(
        results=[
            {
                "id": "memory-1",
                "content": "讨论了一个重要决定",
                "timestamp": reference_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "importance": 0.5,
            }
        ]
    )

    retriever = HybridRetriever(store)
    results = retriever.search("3天前的决定", mode="hybrid", limit=1, time_reference=reference_time)

    assert results[0]["search_mode"] == "hybrid"
    assert isclose(results[0]["alpha"], 0.5, rel_tol=1e-9)
    assert isclose(results[0]["time_weight"], 0.6, rel_tol=1e-9)


def test_hybrid_retriever_uses_cleaned_query_for_time_search():
    store = _FakeMemoryStore(
        results=[
            {
                "id": "memory-1",
                "content": "project kickoff notes",
                "timestamp": "2026-04-01T10:00:00",
                "importance": 0.5,
            }
        ],
        time_range_results=[
            {
                "id": "memory-1",
                "content": "project kickoff notes",
                "timestamp": "2026-04-01T10:00:00",
                "importance": 0.5,
            }
        ],
    )

    retriever = HybridRetriever(store)
    results = retriever.search("project kickoff from last month", mode="auto", limit=1)

    assert len(results) == 1
    assert store.client.last_query == "project kickoff"
