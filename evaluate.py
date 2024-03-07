import matplotlib.pyplot as plt
import json
import os
import numpy as np

def clean_output(data):
    for i in range(len(data)):
        data[i] = data[i].replace(" ", "")
        data[i] = data[i].lower()
    return data

def accuracy(data, text):
    correct = 0
    for output in data:
        if output == text:
            correct += 1
    accuracy = correct / len(data)
    print(accuracy)
    return accuracy

def plot(x, y, line=False):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Fontsize")
    plt.ylabel("Accuracy")
    plt.title("Fontsize vs Accuracy")
    plt.savefig("fontsize_vs_accuracy.png")
    plt.close()

def run(path, fontsize=False, rotation=False):
    accuracies = []
    fontsizes = []
    rotations = []
    for folder in os.listdir(path):
        print(folder)
        for file in os.listdir(f"{path}/{folder}"):
            generated_outputs = []
            if file == "outputs.json":
                with open(f"{path}/{folder}/{file}", "r") as f:
                    data = json.load(f)
                    # print(data)
                for dic in data:
                    output = dic["output"]
                    generated_outputs.append(output)
                # print(generated_outputs)
                # print(len(generated_outputs))

                data = clean_output(generated_outputs)
                acc = accuracy(data)
                accuracies.append(acc)

                if fontsize:
                    fontsize = int(folder.split("_")[2])
                    fontsizes.append(fontsize)

                if rotation:
                    rotation = int(folder.split("_")[1])
                    rotations.append(rotation)

    return accuracies, fontsizes, rotations

if __name__ == "__main__":
    path = "doggos_fontsize/"
    accuracies, fontsizes, _ = run(path, fontsize=True, rotation=False)
    print(accuracies)
    print(fontsizes)
    plot(fontsizes, accuracies)