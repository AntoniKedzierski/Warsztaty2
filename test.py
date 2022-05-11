import numpy as np
from itertools import combinations_with_replacement as cwr

def bondy(degrees):
    for i in range(len(degrees)):
        for j in range(i + 1, len(degrees)):
            if degrees[i] <= i + 1 and degrees[j] < j + 1:
                if degrees[i] + degrees[j] < len(degrees):
                    return False
    return True

def posa(degrees):
    for i in range(len(degrees)):
        if i + 1 <= (len(degrees) - 1) / 2:
            if degrees[i] <= i + 1:
                return False
    if len(degrees) % 2 == 1:
        if degrees[(len(degrees) - 1) / 2] < (len(degrees) + 1) / 2:
            return False
    return True

def chvatal(degrees):
    for i in range(len(degrees)):
        if degrees[i] <= i + 1 < len(degrees) / 2:
            if degrees[len(degrees) - i - 2] < len(degrees) - i - 1:
                return False
    return True

if __name__ == '__main__':
    chvatal([2, 2, 3, 3, 4, 4])
    for p in cwr(range(1, 8), 7):
        if sum(p) % 2 != 0 or max(p) >= len(p) or min(p) == 1:
            continue
        if not bondy(p) and chvatal(p):
            print(p)




