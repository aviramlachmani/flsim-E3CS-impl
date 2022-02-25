import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import ticker
import torch
import math

k = 20      # num of selected clients in each round
K = 100     # num of total activated clients
T = 2500    # num of total rounds

def classA(size):
    return np.random.binomial(size=size, n=1, p=0.1)

def classB(size):
    return np.random.binomial(size=size, n=1, p=0.3)

def classC(size):
    return np.random.binomial(size=size, n=1, p=0.6)

def classD(size):
    return np.random.binomial(size=size, n=1, p=0.9)


def random_n():
    rand_list = []
    out = [0, 0, 0, 0]
    for i in range(20):
        rand_list.append(random.randint(1, 100))
    for rand in rand_list:
        if rand <= 25:
            out[0] += 1
        elif 25 < rand <= 50:
            out[1] += 1
        elif 50 < rand <= 75:
            out[2] += 1
        else:
            out[3] += 1
    return out

def random_d(d, k=20):
    rand_list = []
    out = [0, 0, 0, 0]
    for i in range(d):
        rand_list.append(random.randint(1, 100))
    for rand in rand_list:
        if rand <= 25:
            out[0] += 1
        elif 25 < rand <= 50:
            out[1] += 1
        elif 50 < rand <= 75:
            out[2] += 1
        else:
            out[3] += 1
    pick = k
    for i in range(4):
        if pick == 0:
            out[i] = 0
        elif pick < out[i]:
            out[i] = pick
            pick = 0
        else:
            pick -= out[i]
    return out

def make_CEP_SR_FedCs(T, comm_rounds, k=20):
    cep_sum = np.zeros(T)
    for t in range(T):
        pick = classD(k)
        for x_i_t in pick:
            cep_sum[t] += x_i_t
    CEP_FedCs = sum_up_to_arr(comm_rounds, cep_sum)

    sr_sum = np.zeros(len(comm_rounds))
    for i, T_tag in enumerate(comm_rounds):
        sr_sum[i] = CEP_FedCs[i]/(T_tag*k)
    return CEP_FedCs, sr_sum


def make_CEP_SP(T, comm_rounds, selected_clients_list, k=20):
    cep_sum = np.zeros(T)
    for t in range(T):
        pick = classA(selected_clients_list[0])
        pick = np.append(pick, classB(selected_clients_list[1]))
        pick = np.append(pick, classC(selected_clients_list[2]))
        pick = np.append(pick, classD(selected_clients_list[3]))
        for x_i_t in pick:
            cep_sum[t] += x_i_t
    CEF_res = sum_up_to_arr(comm_rounds, cep_sum)

    SR_sum = np.zeros(len(comm_rounds))
    for i, T_tag in enumerate(comm_rounds):
        SR_sum[i] = CEF_res[i]/(T_tag*k)
    return CEF_res, SR_sum

def make_CEP_SR_E3CS(T, sig_num, sig_type, comm_rounds, K=100):
    Wt = np.ones(K)
    cep_sum = np.zeros(T)

    Xt, At = E3CS_FL_algorithm(k=20, T=T, W_t=Wt, K=K, sig_num=sig_num, sig_type=sig_type)
    for t in range(T):
        for i in At[t]:
            cep_sum[t] += Xt[int(i)]

    CEP_E3CS = sum_up_to_arr(comm_rounds, cep_sum)
    SR_E3CS = np.zeros(len(comm_rounds))
    for i, T_tag in enumerate(comm_rounds):
        SR_E3CS[i] = CEP_E3CS[i] / (T_tag * k)
    return CEP_E3CS, SR_E3CS

def _create_clients_group(K=100, groups=4):
    Xt = []
    group_size = int(K/groups)
    Xt = np.concatenate((classA(group_size), classB(group_size)))
    Xt = np.concatenate((Xt, classC(group_size)))
    Xt = np.concatenate((Xt, classD(group_size)))
    return Xt

def _num_sigma(s_type, num=1):
    def _sigma_t(t):
        return (num*k/K)

    def _inc_sigma_t(t):
        if t<(T/4):
            return 0
        else:
            return k/K

    if s_type=="num":
        return _sigma_t
    else:
        return _inc_sigma_t

def E3CS_FL_algorithm(k, T, W_t, K=100, sig_num=1, sig_type="num", eta=0.5):
    '''
    :param k: the number of involved clients in each round
    :param sig_t: fairness quota
    :param T: final round number
    :param D_i: local data distribution
    :param o1: local update operation
    :param eta: the learning rate of weights update
    :return: - At: the selected group in round t
    '''
    At = np.zeros((T, k))            # default dtype is numpy.float64.
    Pt, St = ([] for i in range(2))
    x_t = _create_clients_group(K)
    print("E3CS-{}({})".format(sig_type, sig_num))
    for t in range(T):
        sigma_t = (_num_sigma(sig_type, sig_num))(t)
        Pt, St = ProbAlloc(k, sigma_t, W_t, K)
        Pt_tensor = torch.tensor(Pt)
        At[t] = torch.multinomial(Pt_tensor, k, replacement=False)
        # At[t] = At[t].detach().numpy()
        selected_clients = [x_t[int(i)] for i in At[t]]
        print("Num of 0 clients: " + str(20-sum(selected_clients)))
        x_estimator_t = np.zeros(K)
        for i in range(0, K):
            x_estimator_t[i] = x_t[i]/Pt[i] if Pt[i]>0.001 else x_t[i]/0.001 # for cases when Pt[i] is very small number
            # x_estimator_t[i] = x_t[i]/Pt[i] if (i in At[t]) else 0
            W_t[i] = W_t[i] if (i in St) else W_t[i]*math.exp((k-(K*sigma_t))*eta*x_estimator_t[i]/K)
    return x_t, At

def ProbAlloc(k, sigma_t, W_t, K=100):
    '''
    :param k: the number of involved clients in each round
    :param sigma_t: fairness quota of round t
    :param W_t: exponential weights for round (vector of size K)
    :param K: total num of activate clients
    :return: - Pt: probability allocation vector for round t
             - St: overflowed set for round t
    '''
    St = []
    P_t = np.zeros(len(W_t))
    for i in range(0, len(W_t)):
        P_t[i] = sigma_t + (((k - (K * sigma_t)) * W_t[i]) / sum(W_t))
        if P_t[i] > 1:
            P_t[i] = 1
            St.append(i)
    P_t = [0 if np.isnan(p) else p for p in P_t]
    return P_t, St

def sum_up_to_arr(T_arr, arr):
    res_arr = np.zeros(len(T_arr))
    for i, t in enumerate(T_arr):
        res_arr[i] = _sum_up_tp(t, arr)
    return res_arr

def _sum_up_tp(T, arr):
    res = 0
    for i in range(T):
       res += arr[i]
    return res

def _aggr_CEP_SR_E3CS(r, T, s_num, s_type, comm_rounds, k=20):
    cep = np.zeros(len(comm_rounds))
    sr = np.zeros(len(comm_rounds))
    for i in range(r):
        cep_tmp, sr_tmp = make_CEP_SR_E3CS(T, s_num, s_type, comm_rounds)
        cep += cep_tmp
        sr += sr_tmp
    CEP_E3CS = (cep / r)
    SR_E3CS = (sr / r)
    return CEP_E3CS, SR_E3CS

def main():
    T = 2500
    r = 10
    dots = 200
    comm_rounds = [i for i in range(1, T, dots)]

    #  make FedCS
    print("FedCS")
    CEP_FedCs, s_r_FedCs = make_CEP_SR_FedCs(T, comm_rounds, k)

    #  make Random
    print("Random")
    random_tmp = random_n()
    CEP_random, s_r_random = make_CEP_SP(T, comm_rounds, random_tmp, k=20)

    #  make pow_d
    d=30
    print("pow_d("+str(d)+")")
    random_tmp_pow = random_d(d, k)
    CEP_pow_d, s_r_pow_d = make_CEP_SP(T, comm_rounds, random_tmp_pow, k=20)

    #  make E3CS-0
    print("E3CS-0")
    CEP_E3CS_0, s_r_E3CS_0 = make_CEP_SR_E3CS(T, 0, "num", comm_rounds)
    CEP_E3CS_0, s_r_E3CS_0 = _aggr_CEP_SR_E3CS(r, T, 0, "num", comm_rounds)

    #  make E3CS-0.5
    print("E3CS-0.5")
    CEP_E3CS_05, s_r_E3CS_05 = make_CEP_SR_E3CS(T, 0.5, "num", comm_rounds)
    CEP_E3CS_05, s_r_E3CS_05 = _aggr_CEP_SR_E3CS(r, T, 0.5, "num", comm_rounds)

    #  make E3CS-0.8
    print("E3CS-0.8")
    CEP_E3CS_08, s_r_E3CS_08 = make_CEP_SR_E3CS(T, 0.8, "num", comm_rounds)
    CEP_E3CS_08, s_r_E3CS_08 = _aggr_CEP_SR_E3CS(r, T, 0.8, "num", comm_rounds)

    #  make E3CS-inc
    print("E3CS-inc")
    CEP_E3CS_inc, s_r_E3CS_inc = make_CEP_SR_E3CS(T, 1, "inc", comm_rounds)
    CEP_E3CS_inc, s_r_E3CS_inc = _aggr_CEP_SR_E3CS(r, T, 1, "inc", comm_rounds)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(comm_rounds, s_r_E3CS_0, label='E3CS-0')
    ax1.plot(comm_rounds, s_r_E3CS_05, label='E3CS-0.5')
    ax1.plot(comm_rounds, s_r_E3CS_08, label='E3CS-0.8')
    ax1.plot(comm_rounds, s_r_E3CS_inc, label='E3CS-inc')
    ax1.plot(comm_rounds, s_r_FedCs, label='FedCS')
    ax1.plot(comm_rounds, s_r_random, label='Random')
    ax1.plot(comm_rounds, s_r_pow_d, label='pow-d')
    ax1.get_yaxis().get_major_formatter().set_useOffset(True)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1, -1))
    ax1.yaxis.major.formatter._useMathText = True
    ax1.set_ylabel('Success Ratio')
    ax1.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax2.plot(comm_rounds, CEP_E3CS_0, label='E3CS-0')
    ax2.plot(comm_rounds, CEP_E3CS_05, label='E3CS-0.5')
    ax2.plot(comm_rounds, CEP_E3CS_08, label='E3CS-0.8')
    ax2.plot(comm_rounds, CEP_E3CS_inc, label='E3CS-inc')
    ax2.plot(comm_rounds, CEP_FedCs, label='FedCS')
    ax2.plot(comm_rounds, CEP_random, label='Random')
    ax2.plot(comm_rounds, CEP_pow_d, label='pow-d')
    ax2.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)
    ax2.get_yaxis().get_major_formatter().set_useOffset(True)
    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('CEP')
    ax2.legend(['E3CS-0', 'E3CS-0.5', 'E3CS-0.8', 'E3CS-inc', 'FedCS', 'Random', 'pow-d'])
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(4, 4))
    ax2.yaxis.major.formatter._useMathText = True
    plt.grid()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
