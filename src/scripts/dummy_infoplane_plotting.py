from src.utils.plotting import plot_multi_trials


def run():
    labels = []
    infos = []
    comps = []
    sizes = []
    colors = []

    big_size = 100
    small_size = 50

    # English topname
    labels.append('English (Topname)')
    infos.append([-0.20])
    comps.append([1.9])
    sizes.append(big_size)
    colors.append('xkcd:blue')

    # English VG Domain.
    labels.append('English (VG Domain)')
    infos.append([-0.28])
    comps.append([1.3])
    sizes.append(big_size)
    colors.append('xkcd:orange')

    # VG Domain Onehot alpha0 1tok

    # VG Domain Onehot alpha10 1tok
    # labels.append('Onehot $\lambda_I=10$ 1tok')
    # infos.append([-0.24415748511011826, -0.2707082140997495, -0.23102903438801425, -0.2478259608705254, -0.2432855808108985])
    # comps.append([1.2929317772388458, 1.1313373565673828, 1.5066378891468049, 1.2865542650222779, 1.2863319158554076])
    # sizes.append(small_size)

    # VG Domain Onehot alpha0 8tok

    # VG Domain Onehot alpha10 8tok
    # labels.append('Onehot $\lambda_I=10$ 8tok')
    # infos.append([-0.10428419045951627, -0.10445041839934482, -0.09896875398192699, -0.10129404553460541, -0.1007218971330522])
    # comps.append([2.7028476119041445, 2.9053736090660096, 3.373599720001221, 3.289308416843414, 2.7828507900238035])
    # sizes.append(small_size)

    # VG Domain VQ alpha0 1tok
    labels.append('VQ $\lambda_I=0; m=1$')
    infos.append([-0.2934157075518464, -0.30414820070888526, -0.2945526030355223, -0.3565649730237216, -0.3070522745375201])
    comps.append([1.2824735045433044, 1.1462082087993621, 1.186035442352295, 1.648440957069397e-08, 1.2101450383663177])
    sizes.append(small_size)
    colors.append('xkcd:green')

    # VG Domain VQ alpha10 1tok
    labels.append('VQ $\lambda_I=10; m=1$')
    infos.append([-0.2126974800471172, -0.20130210551843536, -0.20185996711862397, -0.20526556299410265, -0.20442840864582473])
    comps.append([1.8491870641708374, 1.9638338088989258, 1.9891187310218812, 1.9166909277439117, 1.856919777393341])
    sizes.append(small_size)
    colors.append('xkcd:teal')

    # VG Domain VQ alpha0 8tok
    labels.append('VQ $\lambda_I=0; m=8$')
    infos.append([-0.26058747947482663, -0.27059624024270507, -0.28376338192737993, -0.2712915830951466, -0.2590175496692896])
    comps.append([1.8864408552646637, 1.873309201002121, 1.6679761171340943, 1.768382966518402, 1.832284426689148])
    sizes.append(small_size)
    colors.append('xkcd:salmon')

    # VG Domain VQ alpha10 8tok
    labels.append('VQ $\lambda_I=10; m=8$')
    infos.append([-0.0903385187039786, -0.09125011727314676, -0.09227815894432562, -0.09308867895355644, -0.07773581509145493])
    comps.append([4.76141904592514, 4.154795789718628, 4.694909071922302, 4.042088508605957, 5.582192778587341])
    sizes.append(small_size)
    colors.append('xkcd:magenta')

    plot_multi_trials([comps, infos], labels, sizes, ylabel='Negative MSE', colors=colors, filename='dummyinfoplane.png')


if __name__ == '__main__':
    run()
