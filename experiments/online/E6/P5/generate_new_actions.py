import numpy as np

neutral = np.array([0.398, 0.762, 0.324, 0.000, 0.332, 0.606])

# red1 = np.array([0.000, 0.762, 0.660, 0.000, 1.000, 0.000])
# blue1 = np.array([1.000, 0.762, 0.660, 0.000, 0.000, 0.000])

red2 = np.array([0.000, 0.762, 0.660, 0.000, 0.500, 0.500])
blue2 = np.array([1.000, 1.000, 0.422, 0.000, 0.000, 0.000])

red3 = np.array([0.000, 0.762, 0.660, 0.000, 0.000, 1.000])
blue3 = np.array([1.000, 0.422, 1.000, 0.000, 0.000, 0.000])

red4 = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 1.000])
blue4 = np.array([1.000, 0.000, 0.000, 0.000, 0.000, 0.000])

red5 = np.array([0.000, 0.000, 0.000, 0.000, 1.000, 0.000])

red = np.array([0.000, 0.324, 0.324, 0.000, 1.000, 0.606])
blue = np.array([1.000, 0.324, 0.324, 0.000, 0.000, 0.000])


intensities = np.array([1, 0.675, 1.652])


def main():
    actions = []
    for intensity in intensities:
        actions.append(list(np.round((red * intensity), 5)))
        actions.append(list(np.round((blue * intensity), 5)))
        actions.append(list(np.round((neutral * intensity), 5)))
    print(actions)


if __name__ == "__main__":
    main()
