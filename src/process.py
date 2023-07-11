import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def readTimesFrom():
    times = []
    with open("../log/timesFrom.txt", "r") as f:
        for line in f:
            line = line.split(",")
            for ele in line:
                times.append(float(ele))
    return times

def readTimesTo():
    times = []
    with open("../log/timesTo.txt", "r") as f:
        for line in f:
            line = line.split(",")
            for ele in line:
                times.append(float(ele))
    return times

def main():
    print("Plotting...")
    timesFrom = readTimesFrom()
    timesTo = readTimesTo()

    # use timesFrom and timesTo to draw a density plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(timesFrom, bins=100, density=True, color='blue', alpha=0.5)
    plt.title("From GPU to CPU")
    plt.xlabel("Time (ms)")
    plt.ylabel("Density")
    plt.subplot(1, 2, 2)
    plt.hist(timesTo, bins=100, density=True, color='red', alpha=0.5)
    plt.title("From CPU to GPU")
    plt.xlabel("Time (ms)")
    plt.ylabel("Density")
    plt.savefig("../log/process.png")

if __name__ == "__main__":
    main()
