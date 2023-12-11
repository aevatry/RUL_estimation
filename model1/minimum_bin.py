import math # need the ceil function

# Hyper parameters
L_0_0 = 3000
Ksize_1_0 = 60
Ksize_1_1 = 10
Stride_1_2 = 5
Ksize_2_0 = 40

# Maximum Bin size
L_1_0 = L_0_0 - Ksize_1_0 + 1
L_1_1 = L_1_0 - Ksize_1_1 + 1
L_1_2 = math.ceil(L_1_1/Stride_1_2)
L_2_0 = L_1_2 - Ksize_2_0 + 1

if L_2_0 <= 0:
    print(f"L_2_0 is negative or 0: {L_2_0}. Change Hyper Parameters")

else:
    print(f"L_2_0 is good: {L_2_0}")
    Bin_max = math.floor(math.sqrt(L_2_0))
    print(f"Maximum number of bins is: {Bin_max}")
