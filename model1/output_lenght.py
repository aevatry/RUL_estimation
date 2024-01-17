import math # need the ceil function

def main():

    # Hyper parameters
    Lin = 3600  # so L_0_0/60 minutes of recording necessary
    K1 = 30
    K2 = 20
    S2 = 5
    K3 = 10
    bins = [100, 50, 30]
    F4 = 64

    # Maximum Bin size
    L1 = Lin - K1 + 1
    L2 = math.ceil(((L1-K2)/S2) + 1)
    L3 = L2 - K3 + 1

    print(f"Lenght before pyramid pool: {L3}")
    print(f"lenght of FC entry: {sum(bins)*F4}")


if __name__ == '__main__':
    main()
