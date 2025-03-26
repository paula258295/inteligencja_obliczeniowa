import math

def fct_act(x):
    return 1 / (1 + math.exp(-x))

def forwardPass(wiek, waga, wzrost):
    w1_hidden1, w2_hidden1, w3_hidden1 = -0.46122, 0.97314, -0.39203
    bias_hidden1 = 0.80109

    w1_hidden2, w2_hidden2, w3_hidden2 = 0.78548, 2.10584, -0.57847
    bias_hidden2 = 0.43529

    w_hidden1_output, w_hidden2_output = -0.81546, 1.03775
    bias_output = -0.2368


    hidden1 = wiek * w1_hidden1 + waga * w2_hidden1 + wzrost * w3_hidden1 + bias_hidden1
    hidden1_po_aktywacji = fct_act(hidden1)

    hidden2 = wiek * w1_hidden2 + waga * w2_hidden2 + wzrost * w3_hidden2 + bias_hidden2
    hidden2_po_aktywacji = fct_act(hidden2)

    output = hidden1_po_aktywacji * w_hidden1_output + hidden2_po_aktywacji * w_hidden2_output + bias_output

    return output

result = forwardPass(23, 75, 176)
print(result)  # 0.7985