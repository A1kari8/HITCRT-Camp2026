import os

def process_dir(label_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts and parts[0] == str(1):
                    parts[0] = str(0)
                new_lines.append(' '.join(parts))
            with open(file_path, 'w') as f:
                f.write('\n'.join(new_lines))

process_dir('dataset/train/labels/')
process_dir('dataset/valid/labels/')
