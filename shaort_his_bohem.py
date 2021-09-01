import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    A = np.random.choice([-1, 0, 1], size=(10**6, 6, 6))

    L = np.linalg.eigvals(A).flatten()

    H, x, y = np.histogram2d(L.real, L.imag, bins=500)

    print(1)

    plt.imshow(np.log(H.T+1), extent=[x[0], x[-1], y[0], y[-1]])
    plt.show()
