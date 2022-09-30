import time

import matplotlib.pyplot as plt
import numpy as np

min_range = 0
max_range = 5
sample = 100
step = (max_range - min_range) / sample
xcoor = np.arange(min_range, max_range, step)
ycoor = np.exp(xcoor)
ycoor_act = list(ycoor.copy())

map = np.vstack([xcoor, ycoor]).T
np.random.shuffle(map)
map = map.T

wt = 35
wt_array = np.ones((wt, 1))
xcoor = map[0]
ycoor = map[1]
learning_rate = 0.001
err = 0.0001
cummulative_loss = 0
cummulative_loss_prev = 0
prev_loss = 0
training_dict = {}
ep = 10000
time_list = []
ycoor_pred_list = []


def Win_Discrete(gn):
    wt_vector = wt - gn + 1
    window = np.zeros((wt_vector, wt))
    for i in xcoor:
        index = int(np.floor(wt_vector * ((i - min_range) / (max_range - min_range))))
        training_dict[i] = index
        window[index, index:index + gn] = 1
    return window


def Win_Conti(gn):
    wt_vector = wt - gn + 1
    window = np.zeros((wt_vector, wt))
    kernel = np.linspace(0.1, 1, gn)
    for i in xcoor:
        if gn % 2 == 1:
            kernel[int((len(kernel) + 1) / 2):] = np.flip(kernel[:int((len(kernel) + 1) / 2) - 1])
            index = int(np.floor(wt_vector * ((i - min_range) / (max_range - min_range))))
            training_dict[i] = index
            window[index, index:index + gn] = kernel
        else:
            kernel[int((len(kernel)) / 2):] = np.flip(kernel[:int((len(kernel)) / 2)])
            index = int(np.floor(wt_vector * ((i - min_range) / (max_range - min_range))))
            training_dict[i] = index
            window[index, index:index + gn] = kernel
    return window


def Fit(cummulative_loss, wt_array, gn, string):
    st = time.time()
    ycoor_pred_list = []
    ycoor_prev_list = []
    acc_list = []
    if string.lower() == "discrete":
        window = Win_Discrete(gn)
        fact = 1
    else:
        string = "Continuous"
        window = Win_Conti(gn)
        fact = 10
    for e in range(ep):
        ycoor_prev_list = ycoor_pred_list
        ycoor_pred_list = []
        cummulative_loss_prev = cummulative_loss
        cummulative_loss = 0
        for i, j in zip(map[0, :70], map[1, :70]):
            xcoor_train = training_dict[i]
            xcoor_train = window[xcoor_train, :]
            ycoor_train = j
            ycoor_pred = xcoor_train @ wt_array
            ycoor_pred_list.append(ycoor_pred[0])
            loss = ycoor_train - ycoor_pred
            wt_array_2 = wt_array.copy()
            if len(wt_array_2.shape) == 2:
                wt_array_2 = np.reshape(wt_array_2, (wt_array_2.shape[0]))
            wt_array_2 = wt_array_2 + loss * wt_array_2 * xcoor_train * (learning_rate * fact)
            wt_array_2 = np.reshape(wt_array_2, wt_array.shape)
            wt_array = wt_array_2.copy()
        for i, j in zip(map[0, 70:], map[1, 70:]):
            xcoor_train = training_dict[i]
            xcoor_train = window[xcoor_train, :]
            ycoor_train = j
            ycoor_pred_val = xcoor_train @ wt_array
            ycoor_pred_list.append(ycoor_pred_val[0])
            loss = ycoor_train - ycoor_pred_val
            cummulative_loss += loss

        acc = Acc(ycoor_pred_list, ycoor_act)
        acc_list.append(acc)
        if np.abs(cummulative_loss_prev - cummulative_loss) < (err / fact):
            print('Converging!!!')
            break

    mean_acc = sum(acc_list) / len(acc_list)

    print(f"Mean Accuracy: {mean_acc} for {string} CMAC")
    en = time.time()
    time_req = abs(en - st)
    time_list.append(time_req)
    print(f"Generating Factor: {gn} for {string} CMAC\n")

    xcoor_output = np.array(ycoor_pred_list).reshape(sample)
    map_pred = np.vstack([map[0], xcoor_output])
    map_plot = map_pred.copy().T
    map_plot = map_plot[(map_plot[:, 0]).argsort()]
    map_plot = map_plot.T
    plt.text(2, 125, f"Accuracy: {mean_acc}", fontsize=10)
    plt.plot(map_plot[0], map_plot[1], color='red',label=f"{string} CMAC")

    map_plot = map.copy().T
    map_plot = map_plot[(map_plot[:, 0]).argsort()]
    map_plot = map_plot.T
    plt.plot(map_plot[0], map_plot[1], color='blue', label="Original Map")
    plt.legend()
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    title = "Generating Factor: " + str(gn) + " for " + (string) + " CMAC"
    plt.title(title)
    save = title + '.png'
    plt.savefig(save)
    plt.show()


def Acc(ycoor_pred_list, ycoor_act):
    ycoor_pr = list(ycoor_pred_list)
    ycoor_ac = ycoor_act
    sub = [a - b for a, b in zip(ycoor_pr, ycoor_ac)]
    sq = [s ** 2 for s in sub]
    acc = (1 - (np.sqrt(sum(sq)) / (sample ** 2))) * 100
    return acc


for gn in range(1, wt + 1):
    Fit(cummulative_loss, wt_array, gn, "Discrete")
tim_list = time_list.copy()
time_list = []
plt.plot(range(len(tim_list)), tim_list, color="blue", label="Discrete CMAC")


for gn in range(1, wt + 1):
    Fit(cummulative_loss, wt_array, gn, "Continuous")

plt.plot(range(len(time_list)), time_list, color="red", label="Continuous CMAC")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Generating Factor")
title2 = "Generalization v/s Time"
save2 = title2 + '.png'
plt.title(title2)
plt.savefig(save2)
plt.show()
