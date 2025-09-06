import os

def keep_basketball_labels(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line.startswith('0 ')]
        with open(path, 'w') as f:
            f.writelines(lines)
    return

keep_basketball_labels('dataset2/train/labels')
keep_basketball_labels('dataset2/test/labels')
keep_basketball_labels('dataset2/valid/labels')
