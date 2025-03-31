import json
import time
import datetime
import random
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Callable, Tuple

# Data structures and sample generator
def generate_sample_data(num_records: int) -> List[Dict[str, Any]]:
    """Generate sample hospital patient records"""
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
    
    records = []
    for i in range(num_records):

        # Generate random dates within past 2 years
        days_ago = random.randint(1, 730)
        admission_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Generate random discharge date after admission (some may still be admitted)
        if random.random() > 0.2:  
            discharge_days = random.randint(1, 30)
            discharge_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago-discharge_days)).strftime("%Y-%m-%d")
        else:
            discharge_date = None
        
        record = {
            "patient_id": i + 1000,
            "first_name": random.choice(first_names),
            "last_name": random.choice(last_names),
            "admission_date": admission_date,
            "discharge_date": discharge_date,
            "department": random.choice(["Cardiology", "Neurology", "Oncology", "Pediatrics", "Emergency"]),
            "doctor_id": random.randint(100, 150),
            "room_number": random.randint(100, 500),
            "insurance_id": f"INS-{random.randint(10000, 99999)}"
        }
        records.append(record)
    
    return records

def save_records_to_json(records: List[Dict[str, Any]], filename: str) -> None:
    """Save patient records to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)

def load_records_from_json(filename: str) -> List[Dict[str, Any]]:
    """Load patient records from a JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def get_key_by_patient_id(record: Dict[str, Any]) -> int:
    """Extract patient ID as sorting key"""
    return record["patient_id"]

def get_key_by_name(record: Dict[str, Any]) -> str:
    """Extract patient full name as sorting key"""
    return f"{record['last_name']}, {record['first_name']}"

def get_key_by_admission_date(record: Dict[str, Any]) -> str:
    """Extract admission date as sorting key"""
    return record["admission_date"]

# Bubble Sort Implementation
def bubble_sort(records: List[Dict[str, Any]], key_func: Callable) -> Tuple[List[Dict[str, Any]], float]:
    """
    Apply the Bubble Sort algorithm to sort patient records.
    
    Parameters:
        records: A list of dictionaries, each representing a patient record.
        key_func: A function to extract the sorting key from each record.
        
    Returns:
        A tuple containing the sorted records and the execution time in seconds.
    """
    start_time = time.time()
    result = records.copy()
    n = len(result)
    
    for i in range(n):
        swapped = False
        
        for j in range(0, n - i - 1):
            if key_func(result[j]) > key_func(result[j + 1]):
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        if not swapped:
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time

# Merge Sort Implementation
def merge_sort(records: List[Dict[str, Any]], key_func: Callable) -> Tuple[List[Dict[str, Any]], float]:
    """
    Use the Merge Sort algorithm to sort patient records.
    
    Parameters:
        records: A list of dictionaries, each representing a patient record.
        key_func: A function to extract the sorting key from each record.
        
    Returns:
        A tuple containing the sorted records and the execution time in seconds.
    """
    start_time = time.time()
    result = records.copy()
    

    def _merge_sort(arr):
        if len(arr) <= 1:
            return arr
            
        mid = len(arr) // 2
        left_half = _merge_sort(arr[:mid])
        right_half = _merge_sort(arr[mid:])
        
        return _merge(left_half, right_half)
    
    def _merge(left, right):
        result = []
        i = j = 0

        while i < len(left) and j < len(right):
            if key_func(left[i]) <= key_func(right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    result = _merge_sort(result)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time

def test_both_algorithms(records: List[Dict[str, Any]], key_func: Callable, key_name: str) -> Dict[str, Any]:
    """
    Evaluate the performance of two sorting algorithms on an identical dataset and provide metrics.
    
    Parameters:
        records: A list containing patient records to be sorted.
        key_func: A function used to extract the sorting key.
        key_name: The name of the key being utilized (for reporting purposes).
        
    Returns:
        A dictionary containing performance metrics.
    """
    # Run Bubble Sort
    bubble_sorted, bubble_time = bubble_sort(records, key_func)
    
    # Run Merge Sort
    merge_sorted, merge_time = merge_sort(records, key_func)
    
    if key_func(bubble_sorted[0]) != key_func(merge_sorted[0]) or key_func(bubble_sorted[-1]) != key_func(merge_sorted[-1]):
        print(f"WARNING: Sorting results may differ for key: {key_name}")
    
    return {
        "key": key_name,
        "record_count": len(records),
        "bubble_sort_time": bubble_time,
        "merge_sort_time": merge_time,
        "speedup_factor": bubble_time / merge_time if merge_time > 0 else float('inf')
    }

def run_performance_tests(record_counts: List[int]) -> List[Dict[str, Any]]:
    results = []
    
    for count in record_counts:
        print(f"Testing with {count} records...")
        records = generate_sample_data(count)
 
        id_results = test_both_algorithms(records, get_key_by_patient_id, "patient_id")
        results.append(id_results)
        
        name_results = test_both_algorithms(records, get_key_by_name, "patient_name")
        results.append(name_results)
        
        date_results = test_both_algorithms(records, get_key_by_admission_date, "admission_date")
        results.append(date_results)
        
        if count == max(record_counts):
            bubble_sorted, _ = bubble_sort(records, get_key_by_patient_id)
            merge_sorted, _ = merge_sort(records, get_key_by_patient_id)
            save_records_to_json(bubble_sorted[:10], "bubble_sort_sample.json")
            save_records_to_json(merge_sorted[:10], "merge_sort_sample.json")
    
    return results

def generate_performance_chart(results: List[Dict[str, Any]], output_file: str = "sorting_performance.png") -> None:
    record_counts = sorted(set(result["record_count"] for result in results))
    
    bubble_times = []
    merge_times = []
    
    for count in record_counts:
        count_results = [r for r in results if r["record_count"] == count]
        avg_bubble = sum(r["bubble_sort_time"] for r in count_results) / len(count_results)
        avg_merge = sum(r["merge_sort_time"] for r in count_results) / len(count_results)
        
        bubble_times.append(avg_bubble)
        merge_times.append(avg_merge)
    

    plt.figure(figsize=(10, 6))
    plt.plot(record_counts, bubble_times, 'o-', label='Bubble Sort')
    plt.plot(record_counts, merge_times, 's-', label='Merge Sort')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Records (log scale)')
    plt.ylabel('Execution Time in Seconds (log scale)')
    plt.title('Sorting Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
 
    for i, count in enumerate(record_counts):
        speedup = bubble_times[i] / merge_times[i] if merge_times[i] > 0 else float('inf')
        plt.annotate(f"{speedup:.1f}x", 
                    xy=(count, (bubble_times[i] + merge_times[i]) / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Performance chart saved to {output_file}")

# Main program
def main():
    record_counts = [100, 500, 1000, 5000]
    
    initial_data = generate_sample_data(1000)
    save_records_to_json(initial_data, "hospital_records.json")
    print(f"Generated {len(initial_data)} sample patient records in 'hospital_records.json'")
    
    results = run_performance_tests(record_counts)

    print("\nPerformance Test Results:")
    print("------------------------")
    for result in results:
        print(f"Key: {result['key']}, Records: {result['record_count']}")
        print(f"  Bubble Sort: {result['bubble_sort_time']:.6f} seconds")
        print(f"  Merge Sort:  {result['merge_sort_time']:.6f} seconds")
        print(f"  Speedup:     {result['speedup_factor']:.2f}x")
        print()
    try:
        generate_performance_chart(results)
    except Exception as e:
        print(f"Could not generate chart: {e}")
    
    print("Program completed successfully")

if __name__ == "__main__":
    main()