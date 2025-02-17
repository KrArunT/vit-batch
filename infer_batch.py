import os
import time
import requests
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Folder containing ImageNet subsamples
dataset_folder = 'images'
url = "http://localhost:9085/predictions/vit-model"

headers = {
    'Content-Type': 'application/octet-stream'  # Adjust content type if needed
}

# Function to select random image samples
def get_random_samples(files, sample_size=10):
    return random.sample(files, min(sample_size, len(files)))  # Avoid exceeding list size

# Function to send an HTTP request for inference
def send_request(file_path):
    with open(file_path, 'rb') as f:
        start_time = time.time()
        response = requests.post(url, data=f, headers=headers)
        time_taken = time.time() - start_time
        return {
            'file': os.path.basename(file_path),
            'status_code': response.status_code,
            'prediction': response.json() if response.status_code == 200 else None,
            'time_taken': time_taken
        }

# Function to benchmark the server with concurrent requests
def benchmark_server_concurrent(dataset_folder, num_threads=10):
    files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.JPEG', '.png'))]
    files = get_random_samples(files, sample_size=4)  # Select up to 100 random images

    results = []  # Store results
    total_time_start = time.time()  # Track total time

    # Send requests concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(send_request, file): file for file in files}

        for future in as_completed(future_to_file):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")

    total_time_end = time.time()  # End timing
    total_time = total_time_end - total_time_start
    num_files = len(results)

    # Calculate throughput
    samples_per_second = num_files / total_time if total_time > 0 else 0

    return results, total_time, samples_per_second

# Function to run benchmarking multiple times
def send_data():
    count = 1
    all_metrics = {}

    while count <= 10:  # Run the benchmark 10 times
        benchmark_results, total_time, samples_per_second = benchmark_server_concurrent(dataset_folder, num_threads=10)

        # Convert results to DataFrame
        df = pd.DataFrame(benchmark_results)

        # Print benchmark results and samples per second
        print("\nBenchmark Results:")
        print(df)
        print(f"\nTotal Time Taken: {total_time:.2f} seconds")
        print(f"Samples Per Second: {samples_per_second:.2f} samples/sec")

        # Store metrics
        metrics = {
            f"sample-{count}": [
                f"Total Time Taken: {total_time:.2f} seconds",
                f"Samples Per Second: {samples_per_second:.2f} samples/sec"
            ]
        }
        all_metrics.update(metrics)

        # Save results
        df.to_csv('benchmark_results.csv', index=False)
        df.to_json('benchmark_results.json', indent=4)

        print("\nBenchmark results saved to 'benchmark_results.csv'")
        print(metrics)

        count += 1  # Increase count

    return all_metrics

# Function to publish metrics (to MQTT or other services)
def publish(metrics):
    # Implement MQTT publishing logic here
    pass

if __name__ == "__main__":
    metrics = send_data()
    publish(metrics)
