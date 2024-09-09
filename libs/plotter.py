import numpy as np
import matplotlib.pyplot as plt

def plot_data(ja_tokens, ua_tokens):
    lengths = []
    for token in ja_tokens:
        lengths.append(len(token))
    for token in ua_tokens:
        lengths.append(len(token))
    lengths = np.array(lengths)
    plt.hist(lengths, np.linspace(0, 50, 101))
    plt.ylim(plt.ylim())
    avg_length = lengths.mean()
    plt.plot([avg_length, avg_length], plt.ylim())
    max_length = max(lengths)
    plt.plot([max_length, max_length], plt.ylim())
    plt.title(f'Maximum tokens per example: {max_length} and average tokens per example: {avg_length}')