import os
import re

desired_order = [
    "boolq.txt",
    "piqa.txt",
    "social_i_qa.txt",
    "hellaswag.txt",
    "winogrande.txt",
    "ARC-Easy.txt",
    "ARC-Challenge.txt",
    "openbookqa.txt"
]

output_file = "eval_all.txt"

accuracy_pattern = re.compile(r'accuracy\s+(\d+)\s+([\d.]+)')

with open(output_file, "w") as eval_file:
    current_dir = "./"

    for sub_dir in os.listdir(current_dir):
        sub_dir_path = os.path.join(current_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            for file_name in desired_order:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(sub_dir_path, file_name)

                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        last_line = lines[-5].strip() if lines else "Empty file"
                        match = accuracy_pattern.search(last_line)

                        if match:
                            accuracy_value = float(match.group(2)) * 100
                            accuracy_str = f"{accuracy_value:.2f}"
                            output = f"{sub_dir} + {file_name} + {last_line} | accuracy {accuracy_str}"
                            print(output)
