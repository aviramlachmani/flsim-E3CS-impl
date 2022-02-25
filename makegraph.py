import os.path
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np


def graph():
    # graph EMNIST-Letter, iid, FedAvg-based
    round_t = []
    emnist_random_iid_a = [0] * 400
    emnist_FedCS_iid_a = [0] * 400
    emnist_pow_d_iid_a = [0] * 400
    emnist_E3CS_0_iid_a = [0] * 400
    emnist_E3CS_05_iid_a = [0] * 400
    emnist_E3CS_08_iid_a = [0] * 400
    emnist_E3CS_inc_iid_a = [0] * 400

    if os.path.isfile("output_emnist_random_iid_a.txt"):
        emnist_random_iid_a_file = open("output_emnist_random_iid_a.txt")
        emnist_random_iid_a = []
        with emnist_random_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_random_iid_a.append(float(line[3]) / 100)
                    round_t.append(int(line[1]))
                else:
                    x += 1

    if os.path.isfile("output_emnist_FedCS_iid_a.txt"):
        emnist_FedCS_iid_a_file = open("output_emnist_FedCS_iid_a.txt")
        emnist_FedCS_iid_a = []
        with emnist_FedCS_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_FedCS_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_pow-d_iid_a.txt"):
        emnist_pow_d_iid_a_file = open("output_emnist_pow-d_iid_a.txt")
        emnist_pow_d_iid_a = []
        with emnist_pow_d_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_pow_d_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_0_iid_a.txt"):
        emnist_E3CS_0_iid_a_file = open("output_emnist_E3CS_0_iid_a.txt")
        emnist_E3CS_0_iid_a = []
        with emnist_E3CS_0_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_0_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_05_iid_a.txt"):
        emnist_E3CS_05_iid_a_file = open("output_emnist_E3CS_05_iid_a.txt")
        emnist_E3CS_05_iid_a = []
        with emnist_E3CS_05_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_05_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_08_iid_a.txt"):
        emnist_E3CS_08_iid_a_file = open("output_emnist_E3CS_08_iid_a.txt")
        emnist_E3CS_08_iid_a = []
        with emnist_E3CS_08_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_08_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_inc_iid_a.txt"):
        emnist_E3CS_inc_iid_a_file = open("output_emnist_E3CS_inc_iid_a.txt")
        emnist_E3CS_inc_iid_a = []
        with emnist_E3CS_inc_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_inc_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(1)
    plt.plot(round_t, emnist_E3CS_0_iid_a, "pink")
    plt.plot(round_t, emnist_E3CS_05_iid_a, 'b')
    plt.plot(round_t, emnist_E3CS_08_iid_a, 'c')
    plt.plot(round_t, emnist_E3CS_inc_iid_a, 'g')
    plt.plot(round_t, emnist_FedCS_iid_a, 'y')
    plt.plot(round_t, emnist_random_iid_a, 'orange')
    plt.plot(round_t, emnist_pow_d_iid_a, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("EMNIST-Letter, iid, FedAvg-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph EMNIST-Letter, non-iid, FedAvg-based
    round_t = []
    emnist_random_non_iid_a = [0] * 400
    emnist_FedCS_non_iid_a = [0] * 400
    emnist_pow_d_non_iid_a = [0] * 400
    emnist_E3CS_0_non_iid_a = [0] * 400
    emnist_E3CS_05_non_iid_a = [0] * 400
    emnist_E3CS_08_non_iid_a = [0] * 400
    emnist_E3CS_inc_non_iid_a = [0] * 400

    if os.path.isfile("output_emnist_random_non_iid_a.txt"):
        emnist_random_non_iid_a_file = open("output_emnist_random_non_iid_a.txt")
        emnist_random_non_iid_a = []
        with emnist_random_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_random_non_iid_a.append(float(line[3]) / 100)
                    round_t.append(int(line[1]))
                else:
                    x += 1
    if os.path.isfile("output_emnist_FedCS_non_iid_a.txt"):
        emnist_FedCS_non_iid_a_file = open("output_emnist_FedCS_non_iid_a.txt")
        emnist_FedCS_non_iid_a = []
        with emnist_FedCS_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_FedCS_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_pow-d_non_iid_a.txt"):
        emnist_pow_d_non_iid_a_file = open("output_emnist_pow-d_non_iid_a.txt")
        emnist_pow_d_non_iid_a = []
        with emnist_pow_d_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_pow_d_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_E3CS_0_non_iid_a.txt"):
        emnist_E3CS_0_non_iid_a_file = open("output_emnist_E3CS_0_non_iid_a.txt")
        emnist_E3CS_0_non_iid_a = []
        with emnist_E3CS_0_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_0_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_E3CS_05_non_iid_a.txt"):
        emnist_E3CS_05_non_iid_a_file = open("output_emnist_E3CS_05_non_iid_a.txt")
        emnist_E3CS_05_non_iid_a = []
        with emnist_E3CS_05_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_05_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_E3CS_08_non_iid_a.txt"):
        emnist_E3CS_08_non_iid_a_file = open("output_emnist_E3CS_08_non_iid_a.txt")
        emnist_E3CS_08_non_iid_a = []
        with emnist_E3CS_08_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_08_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_inc_iid_a.txt"):
        emnist_E3CS_inc_non_iid_a_file = open("output_emnist_E3CS_inc_iid_a.txt")
        emnist_E3CS_inc_non_iid_a = []
        with emnist_E3CS_inc_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_inc_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(2)
    plt.plot(round_t, emnist_E3CS_0_non_iid_a, "pink")
    plt.plot(round_t, emnist_E3CS_05_non_iid_a, 'b')
    plt.plot(round_t, emnist_E3CS_08_non_iid_a, 'c')
    plt.plot(round_t, emnist_E3CS_inc_non_iid_a, 'g')
    plt.plot(round_t, emnist_FedCS_non_iid_a, 'y')
    plt.plot(round_t, emnist_random_non_iid_a, 'orange')
    plt.plot(round_t, emnist_pow_d_non_iid_a, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("EMNIST-Letter, non-iid, FedAvg-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph EMNIST-Letter, iid, FedProx-based
    emnist_random_iid_p = [0] * 400
    emnist_FedCS_iid_p = [0] * 400
    emnist_pow_d_iid_p = [0] * 400
    emnist_E3CS_0_iid_p = [0] * 400
    emnist_E3CS_05_iid_p = [0] * 400
    emnist_E3CS_08_iid_p = [0] * 400
    emnist_E3CS_inc_iid_p = [0] * 400

    if os.path.isfile("output_emnist_random_iid_p.txt"):
        emnist_random_iid_p_file = open("output_emnist_random_iid_p.txt")
        emnist_random_iid_p = []
        with emnist_random_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_random_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_FedCS_iid_p.txt"):
        emnist_FedCS_iid_p_file = open("output_emnist_FedCS_iid_p.txt")
        emnist_FedCS_iid_p = []
        with emnist_FedCS_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_FedCS_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_pow-d_iid_p.txt"):
        emnist_pow_d_iid_p_file = open("output_emnist_pow-d_iid_p.txt")
        emnist_pow_d_iid_p = []
        with emnist_pow_d_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_pow_d_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_0_iid_p.txt"):
        emnist_E3CS_0_iid_p_file = open("output_emnist_E3CS_0_iid_p.txt")
        emnist_E3CS_0_iid_p = []
        with emnist_E3CS_0_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_0_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_E3CS_05_iid_p.txt"):
        emnist_E3CS_05_iid_p_file = open("output_emnist_E3CS_05_iid_p.txt")
        emnist_E3CS_05_iid_p = []
        with emnist_E3CS_05_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_05_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_08_iid_p.txt"):
        emnist_E3CS_08_iid_p_file = open("output_emnist_E3CS_08_iid_p.txt")
        emnist_E3CS_08_iid_p = []
        with emnist_E3CS_08_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_08_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_inc_iid_p.txt"):
        emnist_E3CS_inc_iid_p_file = open("output_emnist_E3CS_inc_iid_p.txt")
        emnist_E3CS_inc_iid_p = []
        with emnist_E3CS_inc_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_inc_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(3)
    plt.plot(round_t, emnist_E3CS_0_iid_p, "pink")
    plt.plot(round_t, emnist_E3CS_05_iid_p, 'b')
    plt.plot(round_t, emnist_E3CS_08_iid_p, 'c')
    plt.plot(round_t, emnist_E3CS_inc_iid_p, 'g')
    plt.plot(round_t, emnist_FedCS_iid_p, 'y')
    plt.plot(round_t, emnist_random_iid_p, 'orange')
    plt.plot(round_t, emnist_pow_d_iid_p, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("EMNIST-Letter, iid, FedProx-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph EMNIST-Letter, non-iid, FedProx-based
    emnist_random_non_iid_p = [0] * 400
    emnist_FedCS_non_iid_p = [0] * 400
    emnist_pow_d_non_iid_p = [0] * 400
    emnist_E3CS_0_non_iid_p = [0] * 400
    emnist_E3CS_05_non_iid_p = [0] * 400
    emnist_E3CS_08_non_iid_p = [0] * 400
    emnist_E3CS_inc_non_iid_p = [0] * 400

    if os.path.isfile("output_emnist_random_non_iid_p.txt"):
        emnist_random_non_iid_p_file = open("output_emnist_random_non_iid_p.txt")
        emnist_random_non_iid_p = []
        with emnist_random_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_random_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_FedCS_non_iid_p.txt"):
        emnist_FedCS_non_iid_p_file = open("output_emnist_FedCS_non_iid_p.txt")
        emnist_FedCS_non_iid_p = []
        with emnist_FedCS_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_FedCS_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_pow-d_non_iid_p.txt"):
        emnist_pow_d_non_iid_p_file = open("output_emnist_pow-d_non_iid_p.txt")
        emnist_pow_d_non_iid_p = []
        with emnist_pow_d_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_pow_d_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_emnist_E3CS_0_non_iid_p.txt"):
        emnist_E3CS_0_non_iid_p_file = open("output_emnist_E3CS_0_non_iid_p.txt")
        emnist_E3CS_0_non_iid_p = []
        with emnist_E3CS_0_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_0_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_05_non_iid_p.txt"):
        emnist_E3CS_05_non_iid_p_file = open("output_emnist_E3CS_05_non_iid_p.txt")
        emnist_E3CS_05_non_iid_p = []
        with emnist_E3CS_05_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_05_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_08_non_iid_p.txt"):
        emnist_E3CS_08_non_iid_p_file = open("output_emnist_E3CS_08_non_iid_p.txt")
        emnist_E3CS_08_non_iid_p = []
        with emnist_E3CS_08_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_08_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_emnist_E3CS_inc_non_iid_p.txt"):
        emnist_E3CS_inc_non_iid_p_file = open("output_emnist_E3CS_inc_non_iid_p.txt")
        emnist_E3CS_inc_non_iid_p = []
        with emnist_E3CS_inc_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    emnist_E3CS_inc_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(4)
    plt.plot(round_t, emnist_E3CS_0_non_iid_p, "pink")
    plt.plot(round_t, emnist_E3CS_05_non_iid_p, 'b')
    plt.plot(round_t, emnist_E3CS_08_non_iid_p, 'c')
    plt.plot(round_t, emnist_E3CS_inc_non_iid_p, 'g')
    plt.plot(round_t, emnist_FedCS_non_iid_p, 'y')
    plt.plot(round_t, emnist_random_non_iid_p, 'orange')
    plt.plot(round_t, emnist_pow_d_non_iid_p, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("EMNIST-Letter, non-iid, FedProx-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph cifar, iid, FedAvg-based
    round_t = []
    cifar_random_iid_a = [0] * 200
    cifar_FedCS_iid_a = [0] * 200
    cifar_pow_d_iid_a = [0] * 200
    cifar_E3CS_0_iid_a = [0] * 200
    cifar_E3CS_05_iid_a = [0] * 200
    cifar_E3CS_08_iid_a = [0] * 200
    cifar_E3CS_inc_iid_a = [0] * 200

    if os.path.isfile("output_cifar_random_iid_a.txt"):
        cifar_random_iid_a_file = open("output_cifar_random_iid_a.txt")
        cifar_random_iid_a = []
        with cifar_random_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_random_iid_a.append(float(line[3]) / 100)
                    round_t.append(int(line[1]))
                else:
                    x += 1

    if os.path.isfile("output_cifar_FedCS_iid_a.txt"):
        cifar_FedCS_iid_a_file = open("output_cifar_FedCS_iid_a.txt")
        cifar_FedCS_iid_a = []
        with cifar_FedCS_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_FedCS_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d_iid_a.txt"):
        cifar_pow_d_iid_a_file = open("output_cifar_pow-d_iid_a.txt")
        cifar_pow_d_iid_a = []
        with cifar_pow_d_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_0_iid_a.txt"):
        cifar_E3CS_0_iid_a_file = open("output_cifar_E3CS_0_iid_a.txt")
        cifar_E3CS_0_iid_a = []
        with cifar_E3CS_0_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_0_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_05_iid_a.txt"):
        cifar_E3CS_05_iid_a_file = open("output_cifar_E3CS_05_iid_a.txt")
        cifar_E3CS_05_iid_a = []
        with cifar_E3CS_05_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_05_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_08_iid_a.txt"):
        cifar_E3CS_08_iid_a_file = open("output_cifar_E3CS_08_iid_a.txt")
        cifar_E3CS_08_iid_a = []
        with cifar_E3CS_08_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_08_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_inc_iid_a.txt"):
        cifar_E3CS_inc_iid_a_file = open("output_cifar_E3CS_inc_iid_a.txt")
        cifar_E3CS_inc_iid_a = []
        with cifar_E3CS_inc_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_inc_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(5)
    plt.plot(round_t, cifar_E3CS_0_iid_a, "pink")
    plt.plot(round_t, cifar_E3CS_05_iid_a, 'b')
    plt.plot(round_t, cifar_E3CS_08_iid_a, 'c')
    plt.plot(round_t, cifar_E3CS_inc_iid_a, 'g')
    plt.plot(round_t, cifar_FedCS_iid_a, 'y')
    plt.plot(round_t, cifar_random_iid_a, 'orange')
    plt.plot(round_t, cifar_pow_d_iid_a, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("cifar-10, iid, FedAvg-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph cifar-Letter, non-iid, FedAvg-based
    cifar_random_non_iid_a = [0] * 200
    cifar_FedCS_non_iid_a = [0] * 200
    cifar_pow_d_non_iid_a = [0] * 200
    cifar_E3CS_0_non_iid_a = [0] * 200
    cifar_E3CS_05_non_iid_a = [0] * 200
    cifar_E3CS_08_non_iid_a = [0] * 200
    cifar_E3CS_inc_non_iid_a = [0] * 200

    if os.path.isfile("output_cifar_random_non_iid_a.txt"):
        cifar_random_non_iid_a_file = open("output_cifar_random_non_iid_a.txt")
        cifar_random_non_iid_a = []
        with cifar_random_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_random_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_FedCS_non_iid_a.txt"):
        cifar_FedCS_non_iid_a_file = open("output_cifar_FedCS_non_iid_a.txt")
        cifar_FedCS_non_iid_a = []
        with cifar_FedCS_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_FedCS_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d_non_iid_a.txt"):
        cifar_pow_d_non_iid_a_file = open("output_cifar_pow-d_non_iid_a.txt")
        cifar_pow_d_non_iid_a = []
        with cifar_pow_d_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_0_non_iid_a.txt"):
        cifar_E3CS_0_non_iid_a_file = open("output_cifar_E3CS_0_non_iid_a.txt")
        cifar_E3CS_0_non_iid_a = []
        with cifar_E3CS_0_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_0_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_05_non_iid_a.txt"):
        cifar_E3CS_05_non_iid_a_file = open("output_cifar_E3CS_05_non_iid_a.txt")
        cifar_E3CS_05_non_iid_a = []
        with cifar_E3CS_05_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_05_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_08_non_iid_a.txt"):
        cifar_E3CS_08_non_iid_a_file = open("output_cifar_E3CS_08_non_iid_a.txt")
        cifar_E3CS_08_non_iid_a = []
        with cifar_E3CS_08_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_08_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_inc_non_iid_a.txt"):
        cifar_E3CS_inc_non_iid_a_file = open("output_cifar_E3CS_inc_non_iid_a.txt")
        cifar_E3CS_inc_non_iid_a = []
        with cifar_E3CS_inc_non_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_inc_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(6)
    plt.plot(round_t, cifar_E3CS_0_non_iid_a, "pink")
    plt.plot(round_t, cifar_E3CS_05_non_iid_a, 'b')
    plt.plot(round_t, cifar_E3CS_08_non_iid_a, 'c')
    plt.plot(round_t, cifar_E3CS_inc_non_iid_a, 'g')
    plt.plot(round_t, cifar_FedCS_non_iid_a, 'y')
    plt.plot(round_t, cifar_random_non_iid_a, 'orange')
    plt.plot(round_t, cifar_pow_d_non_iid_a, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("cifar-10, iid, FedAvg-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph cifar, iid, Fedprox-based
    cifar_random_iid_p = [0] * 200
    cifar_FedCS_iid_p = [0] * 200
    cifar_pow_d_iid_p = [0] * 200
    cifar_E3CS_0_iid_p = [0] * 200
    cifar_E3CS_05_iid_p = [0] * 200
    cifar_E3CS_08_iid_p = [0] * 200
    cifar_E3CS_inc_iid_p = [0] * 200

    if os.path.isfile("output_cifar_random_iid_p.txt"):
        cifar_random_iid_p_file = open("output_cifar_random_iid_p.txt")
        cifar_random_iid_p = []
        with cifar_random_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_random_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_FedCS_iid_p.txt"):
        cifar_FedCS_iid_p_file = open("output_cifar_FedCS_iid_p.txt")
        cifar_FedCS_iid_p = []
        with cifar_FedCS_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_FedCS_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d_iid_p.txt"):
        cifar_pow_d_iid_p_file = open("output_cifar_pow-d_iid_p.txt")
        cifar_pow_d_iid_p = []
        with cifar_pow_d_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_0_iid_p.txt"):
        cifar_E3CS_0_iid_p_file = open("output_cifar_E3CS_0_iid_p.txt")
        cifar_E3CS_0_iid_p = []
        with cifar_E3CS_0_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_0_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_05_iid_p.txt"):
        cifar_E3CS_05_iid_p_file = open("output_cifar_E3CS_05_iid_p.txt")
        cifar_E3CS_05_iid_p = []
        with cifar_E3CS_05_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_05_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_08_iid_p.txt"):
        cifar_E3CS_08_iid_p_file = open("output_cifar_E3CS_08_iid_p.txt")
        cifar_E3CS_08_iid_p = []
        with cifar_E3CS_08_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_08_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_inc_iid_p.txt"):
        cifar_E3CS_inc_iid_p_file = open("output_cifar_E3CS_inc_iid_p.txt")
        cifar_E3CS_inc_iid_p = []
        with cifar_E3CS_inc_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_inc_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(7)
    plt.plot(round_t, cifar_E3CS_0_iid_p, "pink")
    plt.plot(round_t, cifar_E3CS_05_iid_p, 'b')
    plt.plot(round_t, cifar_E3CS_08_iid_p, 'c')
    plt.plot(round_t, cifar_E3CS_inc_iid_p, 'g')
    plt.plot(round_t, cifar_FedCS_iid_p, 'y')
    plt.plot(round_t, cifar_random_iid_p, 'orange')
    plt.plot(round_t, cifar_pow_d_iid_p, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("cifar-10, iid, FedProx-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph cifar-Letter, non-iid, FedProx-based
    cifar_random_non_iid_p = [0] * 200
    cifar_FedCS_non_iid_p = [0] * 200
    cifar_pow_d_non_iid_p = [0] * 200
    cifar_E3CS_0_non_iid_p = [0] * 200
    cifar_E3CS_05_non_iid_p = [0] * 200
    cifar_E3CS_08_non_iid_p = [0] * 200
    cifar_E3CS_inc_non_iid_p = [0] * 200

    if os.path.isfile("output_cifar_random_non_iid_p.txt"):
        cifar_random_non_iid_p_file = open("output_cifar_random_non_iid_p.txt")
        cifar_random_non_iid_p = []
        with cifar_random_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_random_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_FedCS_non_iid_p.txt"):
        cifar_FedCS_non_iid_p_file = open("output_cifar_FedCS_non_iid_p.txt")
        cifar_FedCS_non_iid_p = []
        with cifar_FedCS_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_FedCS_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d_non_iid_p.txt"):
        cifar_pow_d_non_iid_p_file = open("output_cifar_pow-d_non_iid_p.txt")
        cifar_pow_d_non_iid_p = []
        with cifar_pow_d_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_0_non_iid_p.txt"):
        cifar_E3CS_0_non_iid_p_file = open("output_cifar_E3CS_0_non_iid_p.txt")
        cifar_E3CS_0_non_iid_p = []
        with cifar_E3CS_0_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_0_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_05_non_iid_p.txt"):
        cifar_E3CS_05_non_iid_p_file = open("output_cifar_E3CS_05_non_iid_p.txt")
        cifar_E3CS_05_non_iid_p = []
        with cifar_E3CS_05_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_05_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_E3CS_08_non_iid_p.txt"):
        cifar_E3CS_08_non_iid_p_file = open("output_cifar_E3CS_08_non_iid_p.txt")
        cifar_E3CS_08_non_iid_p = []
        with cifar_E3CS_08_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_08_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1
    if os.path.isfile("output_cifar_E3CS_inc_non_iid_p.txt"):
        cifar_E3CS_inc_non_iid_p_file = open("output_cifar_E3CS_inc_non_iid_p.txt")
        cifar_E3CS_inc_non_iid_p = []
        with cifar_E3CS_inc_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_E3CS_inc_non_iid_p.append(float(line[3]) / 100)
                else:
                    x += 1

    # make grahp
    plt.figure(8)
    plt.plot(round_t, cifar_E3CS_0_non_iid_p, "pink")
    plt.plot(round_t, cifar_E3CS_05_non_iid_p, 'b')
    plt.plot(round_t, cifar_E3CS_08_non_iid_p, 'c')
    plt.plot(round_t, cifar_E3CS_inc_non_iid_p, 'g')
    plt.plot(round_t, cifar_FedCS_non_iid_p, 'y')
    plt.plot(round_t, cifar_random_non_iid_p, 'orange')
    plt.plot(round_t, cifar_pow_d_non_iid_p, 'r')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("cifar-10, non-iid, FedProx-based")
    plt.legend(["E3CS-0", "E3CS-05", "E3CS-08", "E3CS-inc", "FedCS", "Random", "pow-d"])
    plt.show()

    # graph cifar-Letter, non-iid, FedProx-based
    cifar_pow_d_30_non_iid_a = [0] * 200
    cifar_pow_d_50_non_iid_a = [0] * 200
    cifar_pow_d_70_non_iid_a = [0] * 200

    if os.path.isfile("output_cifar_pow-d=30_iid_a.txt"):
        cifar_pow_d_30_non_iid_p_file = open("output_cifar_pow-d=30_iid_a.txt")
        cifar_pow_d_30_non_iid_a = []
        with cifar_pow_d_30_non_iid_p_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_30_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d=50_iid_a.txt"):
        cifar_pow_d_50_iid_a_file = open("output_cifar_pow-d=50_iid_a.txt")
        cifar_pow_d_50_non_iid_a = []
        with cifar_pow_d_50_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_50_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1

    if os.path.isfile("output_cifar_pow-d=70_iid_a.txt"):
        cifar_pow_d_70_iid_a_file = open("output_cifar_pow-d=70_iid_a.txt")
        cifar_pow_d_70_non_iid_a = []
        with cifar_pow_d_50_iid_a_file as file:
            lines = file.readlines()
            x = 0
            for line in lines:
                if x == 1:
                    line = line.split()
                    cifar_pow_d_70_non_iid_a.append(float(line[3]) / 100)
                else:
                    x += 1
    # make grahp
    plt.figure(9)
    plt.plot(round_t, cifar_pow_d_30_non_iid_a, "pink")
    plt.plot(round_t, cifar_pow_d_50_non_iid_a, 'b')
    plt.plot(round_t, cifar_pow_d_70_non_iid_a, 'c')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.title("cifar-10, iid, FedAvg-based ")
    plt.legend(["pow-d=30", "pow-d=50", "pow-d=70"])
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    graph()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
