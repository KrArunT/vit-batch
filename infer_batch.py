#!/usr/bin/env python3
import argparse
import os
import time
import random
import json
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pika

# -------------------- RabbitMQ Configuration --------------------
RABBITMQ_HOST = '10.86.16.33'
RABBITMQ_USER = 'admin'
RABBITMQ_PASSWORD = 'Infobell1234#'
CREDENTIALS = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASSWORD)
CONNECTION_PARAMS = pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=CREDENTIALS)

# -------------------- Global Configurations --------------------
# Folder containing your ImageNet subsamples
dataset_folder = Path("images")
# Default header for image requests
headers = {'Content-Type': 'application/octet-stream'}

# -------------------- Argument Parser --------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark a prediction server using an image dataset.')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of image samples to process in each benchmark run.')
    parser.add_argument('--port', type=str, default='8080',
                        help='Port number for the prediction server.')
    parser.add_argument('--hostname', type=str, required=True,
                        help='Hostname or IP address of the prediction server.')
    parser.add_argument('--topic', type=str, required=True,
                        help='RabbitMQ queue (topic) to publish metrics.')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of benchmarking iterations to run (set to 0 for infinite loop).')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent threads for sending requests.')
    return parser.parse_args()

# -------------------- Helper Functions --------------------
def get_random_samples(files, sample_size):
    """Return a random sample of files, not exceeding the available count."""
    return random.sample(files, min(sample_size, len(files)))

def send_request(file_path, url):
    """Send an image file for inference and return timing and status details."""
    with open(file_path, 'rb') as f:
        start_time = time.time()
        try:
            response = requests.post(url, data=f, headers=headers)
            response.raise_for_status()
        except Exception as e:
            return {
                'file': os.path.basename(file_path),
                'status_code': getattr(response, 'status_code', None),
                'prediction': None,
                'time_taken': time.time() - start_time,
                'error': str(e)
            }
        time_taken = time.time() - start_time
        try:
            pred = response.json() if response.status_code == 200 else None
        except Exception:
            pred = None
        return {
            'file': os.path.basename(file_path),
            'status_code': response.status_code,
            'prediction': pred,
            'time_taken': time_taken
        }

def benchmark_server_concurrent(dataset_folder, sample_size, url, num_threads):
    """
    Benchmark the server by sending a random sample of images concurrently.
    Returns detailed results, total elapsed time, and computed throughput.
    """
    # Collect image files (considering common image file extensions)
    files = [str(f) for f in dataset_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    files = get_random_samples(files, sample_size)

    results = []
    total_time_start = time.time()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {executor.submit(send_request, file, url): file for file in files}
        for future in as_completed(future_to_file):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")

    total_time = time.time() - total_time_start
    num_files = len(results)
    samples_per_second = num_files / total_time if total_time > 0 else 0

    return results, total_time, samples_per_second

def publish(queue, message):
    """
    Publish a message (as a JSON string) to a specified RabbitMQ queue.
    The message is set to be persistent.
    """
    connection = pika.BlockingConnection(CONNECTION_PARAMS)
    channel = connection.channel()
    # Ensure the queue exists and is durable
    channel.queue_declare(queue=queue, durable=True)
    channel.basic_publish(exchange='',
                          routing_key=queue,
                          body=message,
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))
    print(f"Sent to '{queue}': {message}")
    connection.close()

# -------------------- Main Benchmark Loop --------------------
def send_data(sample_size, port, hostname, topic, iterations, concurrency):
    """
    Run the benchmark for a given number of iterations (or indefinitely if iterations==0)
    and publish the throughput metrics to RabbitMQ.
    """
    url = f"http://{hostname}:{port}/predictions/vit-model"
    iteration = 1
    while iterations == 0 or iteration <= iterations:
        print(f"\n--- Benchmark Iteration {iteration} ---")
        benchmark_results, total_time, samples_per_second = benchmark_server_concurrent(
            dataset_folder, sample_size, url, concurrency
        )

        # Save results to CSV and JSON files (one set per iteration)
        df = pd.DataFrame(benchmark_results)
        csv_file = f'benchmark_results_{iteration}.csv'
        json_file = f'benchmark_results_{iteration}.json'
        df.to_csv(csv_file, index=False)
        df.to_json(json_file, indent=4)
        print(f"Results saved to '{csv_file}' and '{json_file}'")

        # Build and publish metrics
        metrics = {
            "iteration": iteration,
            "total_time_seconds": round(total_time, 2),
            "samples_per_second": round(samples_per_second, 2)
        }
        publish(topic, json.dumps(metrics))

        print(f"Iteration {iteration}: Total Time = {total_time:.2f}s, Throughput = {samples_per_second:.2f} samples/sec")
        iteration += 1

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    args = parse_args()
    send_data(args.sample_size, args.port, args.hostname, args.topic, args.iterations, args.concurrency)
