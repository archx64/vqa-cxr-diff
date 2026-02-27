from tqdm import tqdm
import time, random
from colorama import Fore, Back, Style

def main():
    print(Fore.LIGHTGREEN_EX, Back.BLACK)
    epoches = 1000
    total = 0
    i = 0
    for epoch  in range(epoches):
        for _ in tqdm(range(1000), desc=f"iteration:{i} {epoch+1} / {epoches}", leave=False):
            time.sleep(0.01)
            i = random.randint(1, 1000)
            

if __name__ == '__main__':
    random.seed(69)
    main()