import math # need the ceil function

def main():

    # Hyper parameters
    L_0_0 = 3600  # so L_0_0/60 minutes of recording necessary
    Ksize_1_0 = 30
    Ksize_1_1 = 1
    Stride_1_2 = 2
    Ksize_2_0 = 10

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

if __name__ == '__main__':
    main()
