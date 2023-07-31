import numpy as np
def get_num(text):
    f1 = text.split(":")[-1]
    f1 = float(f1.strip())
    return f1

def process(dataset):
    with open("./{}_result.txt".format(dataset),"r") as f:
        lines = f.readlines()
        for i in range(0,len(lines),10):
            print(lines[i].strip())
            if i+9 <= len(lines):
                f1 = []
                for j in [1,3,5,7,9]:
                    text = lines[i+j].strip()
                    f1.append(get_num(text))
                mean = np.mean(f1) * 100
                std = np.std(f1) * 100
                print(f'{mean:.2f} $\pm$ {std:.2f}')
if __name__ == '__main__':
    # process('ace')
    # process('maven')
    # process('fewevent')
    process('ere')