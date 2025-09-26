from concurrent.futures import ProcessPoolExecutor
import queue
import subprocess
import os
import sys

model_name = sys.argv[1]
save_dir = model_name.replace("results", "evaluation")

def evaluate(dataset, gpu):
    print('*******dataset:', dataset)
    print('*******model_name:', model_name)
    # model_name = "./results/commonsense15k/Llama-3.2-1B/lora_rank-32"
    # save_dir= "./evaluation/commonsense15k/Llama-3.2-1B/lora_rank-32"
    
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass

    save_path = os.path.join(save_dir, dataset + ".txt")
    command = f"CUDA_VISIBLE_DEVICES={gpu} python commonsense_evaluate.py \
               --dataset {dataset} \
               --base_model '{model_name}' \
               --batch_size 64| tee -a {save_path}"

    result = subprocess.run(command, shell=True, text=True, capture_output=False)
    print(f"Evaluation results for dataset {dataset} on GPU {gpu}:\n{result.stdout}")
    return gpu


datasets = ["boolq", "social_i_qa", "piqa", "ARC-Easy", "ARC-Challenge", "winogrande", "openbookqa", "hellaswag"]

gpus = [0]
tasks_queue = queue.Queue()
gpu_queue = queue.Queue()

for gpu in gpus:
    gpu_queue.put(gpu)
for task in datasets:
    tasks_queue.put(task)

num_processes = min(len(datasets), len(gpus))  # number of processes to run in parallel

with ProcessPoolExecutor(max_workers=num_processes) as executor:
    futures = [executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()) for i in range(num_processes)]
    for future in futures:
        gpu_id = future.result()
        gpu_queue.put(gpu_id)
        if tasks_queue.qsize() > 0:
            futures.append(executor.submit(evaluate, tasks_queue.get(), gpu_queue.get()))
