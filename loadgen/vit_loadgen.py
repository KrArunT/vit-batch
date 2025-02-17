                            routing_key=queue,
                            body=message,
                            properties=pika.BasicProperties(
                                delivery_mode=2,  # Make message persistent
                            ))
    print(f"Sent to {queue}: {message}")

    # Close connection
    connection.close()

def send_data(sample_size, port, hostname, topic):
    url = f"http://{hostname}:{port}/predictions/vit-base-patch16-224"

    count = 1
    while True:  # The loop will run indefinitely unless manually stopped
        # Run the benchmark
        benchmark_results, total_time, samples_per_second = benchmark_server(dataset_folder, sample_size, url)

        # Convert results to DataFrame
        df = pd.DataFrame(benchmark_results)

        #metrics = {"samples_per_second": str(round(samples_per_second,2)) + " samples/sec"}
        metrics = {"samples_per_second": round(samples_per_second,2)}

        count += 1
        publish(f"{topic}", json.dumps(metrics))  # Using JSON for better structure in the message

if __name__ == "__main__":
    args = parse_args()
    send_data(args.sample_size, args.port, args.hostname, args.topic)
                 
