import torch


def compute(x):
    return w * x


def cost(xs, ys):
    sum = 0
    for x, y in zip(xs, ys):
        sum += (compute(x) - y) ** 2
    return sum


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * w * (w * x - y)
    return grad / len(xs)


# w = 1.0
# xs = [112.0,3.0,4.0]
# ys= [2.0,4.0,3.0]
# print('f(4) before training:'+str(compute(4)))
# for i in range(1000):
#     temp_cost = cost(xs,ys)
#     temp_Gradient = gradient(xs,ys)
#     w -= 0.005 * temp_Gradient
#     print('cost:'+str(temp_cost)+' '+'w:'+str(w))
# print('f(4) after training' +compute(4))

if __name__ == '__main__':
    w = 1.0
    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]
    print('f(4) before training:' + str(compute(4)))
    for i in range(1000):
        temp_cost = cost(xs, ys)
        temp_Gradient = gradient(xs, ys)
        w -= 0.005 * temp_Gradient
        print('cost:' + str(temp_cost) + ' ' + 'w:' + str(w))
    print('f(4) after training' + str(compute(2.0)))
    print(torch.__version__)
