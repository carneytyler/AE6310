# Homework 2 Problem 3
# Tyler Carney

import numpy as np
import sympy as sp
from sympy.solvers.solveset import solveset_real
from math import isclose
import matplotlib.pylab as plt

theta = np.linspace(0, 2*np.pi, 100)

def modelFun(x1, x2, B, fun, gradFun):
    p = np.array([[x1], [x2]])
    return fun(x1, x2) + np.dot(np.transpose(gradFun(x1, x2)), p) \
            + .5 * np.dot(np.dot(np.transpose(p), B), p)

def cauchy_step(g, B, delta):
    # Calculates the Cauchy step
    eigh = np.linalg.eig(B)

    gTBg = np.dot(np.dot(np.transpose(g), B), g)

    if gTBg <= 0:
        tau = 1
    else:
        tau = (np.linalg.norm(g, 2) ** 3) / (delta * gTBg)

        if tau > 1.:
            tau = 1.

    return -(tau * delta / np.linalg.norm(g, 2)) * g

def trust_region_step(g, B, delta):
    # Calculates the trust region 
    lamb = sp.Symbol('lamb')

    eigh = np.linalg.eig(B)

    # B is positive definite
    if eigh[0][0] > 0 and eigh[0][1] > 0:
        pMin = -np.linalg.solve(B, g)
        pMin_norm = np.linalg.norm(pMin, 2)

        if pMin_norm <= delta:
            return pMin
    
    # B is not positive definite
    q1T = np.transpose(eigh[1][:, 0])
    q2T = np.transpose(eigh[1][:, 1])

    # Setting up the equation
    pLambda = ((np.dot(q1T, g) ** 2) / ((eigh[0][0] + lamb) ** 2)) \
            + ((np.dot(q2T, g) ** 2) / ((eigh[0][1] + lamb) ** 2)) \
            - delta ** 2

    # Symbolically solving
    lamb_solved = solveset_real(pLambda[0], lamb)
    lamb_solved = list(lamb_solved)

    # Getting the lambda
    for i in range(len(lamb_solved)):
        # Lambda needs to be strictly positive and greater than negative of smallest eig
        if lamb_solved[i] > 0 and lamb_solved[i] > -min(eigh[0]):
            lamb_pos = lamb_solved[i]

    BlambI = B + lamb_pos * np.identity(2)
    BlambI = BlambI.astype(np.float)
    pMin = np.linalg.solve(BlambI, -g)

    return pMin

def functionA(x1, x2):
    return -10*x1**2 + 10*x2**2 + 4*np.sin(x1*x2) - 2*x1 + x1**4

def gradFunA(x1, x2):
    wrt_x1 = -20*x1 + 4*x2*np.cos(x1*x2) - 2 + 4*x1**4
    wrt_x2 = 20*x2 + 4*x1*np.cos(x1*x2)
    return np.array([[wrt_x1], [wrt_x2]])

def functionB(x1, x2):
    return 100*(x2-x1**2)**2 + (1-x1)**2

def gradFunB(x1, x2):
    wrt_x1 = 2*(200*x1**3 - 200*x1*x2 + x1 - 1)
    wrt_x2 = 200*(x2 - x1**2)
    return np.array([[wrt_x1], [wrt_x2]])

def minimize(x1k, x2k, fun, gradFun, approach): 
    #Trust region minimization
    eta = 0
    eps = 1*10**-3

    #Setting initial trust region and max trust region
    deltak = 1
    delta_max = 10

    #Priming the while loop
    gk = gradFun(x1k, x2k)
    Bk = np.array([[1, 0], [0, 1]])

    cntr = 0

    while np.linalg.norm(gk, 2) > eps:
        plt.plot(x1k, x2k, 'ko')
        del1 = x1k + deltak * np.cos(theta)
        del2 = x2k + deltak * np.sin(theta)
        plt.plot(del1, del2, color = "red")
        cntr += 1
        print('-------------------Loop ', cntr, '-------------------')

        gk = gradFun(x1k, x2k)
        print('gradient norm: ', np.linalg.norm(gk, 2))

        if approach == 'cauchy':
            p = cauchy_step(gk, Bk, deltak)
        else:
            p = trust_region_step(gk, Bk, deltak)

        print('step: ', p)

        fk = fun(x1k, x2k)
        fk1 = fun(x1k + p[0][0], x2k + p[1][0])
        mk = modelFun(x1k, x2k, Bk, fun, gradFun)
        mk1 = modelFun(x1k + p[0][0], x2k + p[1][0], Bk, fun, gradFun)

        print('model k: ', mk)
        print('model k1: ', mk1)
        print('function 1: ', fk)
        print('function k1: ', fk1)
        rhok = (fk - fk1) / (mk - mk1)
        
        gk1 = gradFun(x1k + p[0][0], x2k + p[1][0])
        yk = gk1 - gk
        sk = p
        Bk = Bk + (np.dot((yk - np.dot(Bk, sk)), np.transpose((yk - np.dot(Bk, sk)))) \
            / np.dot(np.transpose(sk), (yk - np.dot(Bk, sk))))

        if rhok >= eta:
            print('rho greater than eta (', rhok[0][0], ')')
            x1k += p[0][0]
            x2k += p[1][0]

        print('x1k: ', x1k)
        print('x2k: ', x2k)

        if rhok < .25:
            # print('rho less than .25')
            deltak = deltak * .25
        elif rhok > .75 and isclose(np.linalg.norm(p, 2), deltak):
            # print('rho greater than .75 and equal to delta')
            deltak = np.min(2*deltak, delta_max)

        if cntr > 30:
            break

    return np.array([[x1k], [x2k]])


print(minimize(.5, .5, functionB, gradFunB, 'cauchy'))
plt.show()