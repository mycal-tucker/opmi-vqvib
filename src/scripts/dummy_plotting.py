import matplotlib.pyplot as plt
import math


def run():
    plt.figure(figsize=(4, 2))
    for N in [5, 10, 20, 30]:
        x = []
        y = []
        for kec in range(1, 50):
            x.append(kec)
            num_covered = 0
            for i in range(1, kec + 1):
                dni_sum = 0
                for j in range(0, i + 1):
                    dni_sum += ((-1) ** (i - j)) * math.comb(i, j) * (j ** N)
                dni_sum = dni_sum * math.comb(kec, i)
                prob = dni_sum / (kec ** N)
                num_covered += i * prob
            frac_covered = num_covered / kec
            print("Expected frac", frac_covered)
            y.append(frac_covered)
        plt.plot(x, y, label='N = ' + str(N))

    plt.xlabel('k')
    plt.ylabel('E[s]')
    plt.ylim(0, 1.05)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('/home/mycal/translationcoverage.png')
    plt.show()


if __name__ == '__main__':
    run()