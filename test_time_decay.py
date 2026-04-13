#!/usr/bin/env python3

import sys
import os
from datetime import datetime, timezone, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openclaw_memory.core.hybrid_retrieval import TimeDecay, QueryIntentRecognizer

def test_time_decay_with_reference_time():
    """Test that TimeDecay works with reference time"""
    print("Testing TimeDecay with reference time...")
    
    # Create a TimeDecay instance
    decay = TimeDecay()
    
    # Test timestamp (yesterday)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    timestamp_str = yesterday.strftime("%Y-%m-%dT%H:%M:%S")
    mid_reference = yesterday + timedelta(days=1, hours=12)
    later_reference = yesterday + timedelta(days=3)
    
    # Calculate weight using a fixed in-between reference time
    weight_current = decay.calculate_weight(timestamp_str, 0.5, reference_time=mid_reference)
    
    # Calculate weight using yesterday as reference (should be boosted)
    weight_yesterday_ref = decay.calculate_weight(timestamp_str, 0.5, reference_time=yesterday)
    
    # Calculate weight using a later reference (should be more decayed)
    weight_today_ref = decay.calculate_weight(timestamp_str, 0.5, reference_time=later_reference)
    
    print(f"  Timestamp: {timestamp_str}")
    print(f"  Weight with current time: {weight_current:.3f}")
    print(f"  Weight with yesterday as reference: {weight_yesterday_ref:.3f}")
    print(f"  Weight with today as reference: {weight_today_ref:.3f}")
    
    # The weight should be highest when reference time equals timestamp time
    assert weight_yesterday_ref > weight_current, "Weight should be higher when reference time matches timestamp"
    assert weight_today_ref < weight_current, "Weight should be lower when reference time is after timestamp"
    print("  ✓ TimeDecay reference time test passed")

def test_query_intent_recognizer_time_parsing():
    """Test that QueryIntentRecognizer parses various time references"""
    print("\nTesting QueryIntentRecognizer time parsing...")
    
    recognizer = QueryIntentRecognizer()
    now = datetime.now(timezone.utc)
    
    test_cases = [
        ("今天发生了什么", 0),  # Today
        ("昨天的内容", -1),   # Yesterday
        ("前天发生的事", -2), # Day before yesterday
        ("3天前的决定", -3),  # 3 days ago
        ("1周前的会议", -7),  # 1 week ago
        ("2周前的讨论", -14), # 2 weeks ago
        ("1个月前的计划", -30), # 1 month ago (approx)
        ("2个月前的决策", -60), # 2 months ago (approx)
        ("半年前的项目", -180), # Half year ago (approx)
        ("1年前的想法", -365), # 1 year ago (approx)
        ("What happened today", 0),  # Today
        ("Notes from yesterday", -1),  # Yesterday
        ("What did we decide 3 days ago", -3),  # 3 days ago
        ("The meeting 2 weeks ago", -14),  # 2 weeks ago
        ("The roadmap from 6 months ago", -180),  # Half year equivalent
        ("Ideas from 1 year ago", -365),  # 1 year ago
    ]
    
    for query, expected_days_offset in test_cases:
        result = recognizer.recognize(query)
        reference_time = result.get('reference_time')
        
        if reference_time is not None:
            # Calculate expected date
            expected_date = now + timedelta(days=expected_days_offset)
            expected_date = expected_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate actual offset
            actual_offset = (reference_time - expected_date).days
            
            print(f"  Query: '{query}'")
            print(f"    Reference time: {reference_time.strftime('%Y-%m-%d')}")
            print(f"    Expected: {expected_date.strftime('%Y-%m-%d')} (offset: {expected_days_offset} days)")
            print(f"    Actual offset: {actual_offset} days")
            
            # Allow some flexibility for month/year approximations
            if any(token in query.lower() for token in ["month", "months", "year", "years"]) or "个月" in query or "年前" in query:
                # For month/year calculations, allow larger tolerance due to approximation
                assert abs(actual_offset) <= 15, f"Date offset too large for '{query}'"
            else:
                # For day/week calculations, expect exact match
                assert actual_offset == 0, f"Date mismatch for '{query}'"
            
            print(f"    ✓ Passed")
        else:
            print(f"  Query: '{query}' -> No reference time extracted")
    
    print("  ✓ QueryIntentRecognizer time parsing test passed")

def test_integration():
    """Test the integration between components"""
    print("\nTesting integration...")
    
    # This would normally require a memory store, but we can test the logic
    recognizer = QueryIntentRecognizer()
    decay = TimeDecay()
    
    test_queries = [
        "3天前我们讨论了什么",
        "半年前决定的技术方案",
        "上周的会议纪要",
        "What did we discuss 3 days ago?",
        "The plan from last month"
    ]
    
    for query in test_queries:
        result = recognizer.recognize(query)
        reference_time = result.get('reference_time')
        
        if reference_time:
            # Test that we can calculate weight with this reference time
            # Use a timestamp from the reference time period
            test_timestamp = reference_time.strftime("%Y-%m-%dT%H:%M:%S")
            weight = decay.calculate_weight(test_timestamp, 0.5, reference_time=reference_time)
            
            print(f"  Query: '{query}'")
            print(f"    Reference time: {reference_time.strftime('%Y-%m-%d')}")
            print(f"    Test weight: {weight:.3f}")
            assert weight > 0, "Weight should be positive"
            print(f"    ✓ Integration test passed")
        else:
            print(f"  Query: '{query}' -> No reference time")

if __name__ == "__main__":
    print("Running time decay and intent recognition tests...\n")
    
    try:
        test_time_decay_with_reference_time()
        test_query_intent_recognizer_time_parsing()
        test_integration()
        
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
