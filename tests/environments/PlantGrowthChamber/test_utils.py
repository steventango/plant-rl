import numpy as np
from PIL import Image

from environments.PlantGrowthChamber.utils import process_image
from environments.PlantGrowthChamber.zones import Rect, Tray


def test_process_image():
    areas = []
    for file in ["z3c1--2022-12-31--08-50-01.png", "z3c1--2022-12-31--09-00-01.png"]:
        image = np.array(Image.open(f"tests/test_data/{file}"))
        debug_images = {}
        trays = [
            Tray(
                num_plants=24,
                n_wide=8,
                n_tall=3,
                rect=Rect(
                    top_left=(112, 404),
                    top_right=(1718, 198),
                    bottom_left=(114, 990),
                    bottom_right=(1814, 838),
                    # top_left=(116, 403),
                    # top_right=(1717, 196),
                    # bottom_left=(119, 988),
                    # bottom_right=(1815, 844),
                ),
            ),
            Tray(
                num_plants=24,
                n_wide=8,
                n_tall=3,
                rect=Rect(
                    top_left=(106, 1062),
                    top_right=(1804, 922),
                    bottom_left=(188, 1652),
                    bottom_right=(1832, 1594),
                    # top_left=(111, 1062),
                    # top_right=(1804, 926),
                    # bottom_left=(192, 1651),
                    # bottom_right=(1829, 1590),
                ),
            ),
        ]
        df = process_image(image, trays, debug_images)
        df.to_csv(f"tests/test_data/{file}.csv", index=False)
        areas.append(df["area"])
        for key, value in debug_images.items():
            value.save(f"tests/test_data/{file}_{key}.png")

    # create a side by side barplot comparision of the areas from each file
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(areas[0])), areas[0], color="b", label="2022-12-31--08-50-01")
    ax.bar(np.arange(len(areas[1])) + 0.4, areas[1], color="r", label="2022-12-31--09-00-01")
    ax.legend()
    plt.savefig("tests/test_data/areas.png")


def test_alg():
    from PIL import Image
    from plantcv import plantcv as pcv

    gray_img = np.array(Image.open("tests/test_data/z3c1--2022-12-31--08-50-01.png"))[:, :, 1]

    bin_gauss1 = pcv.threshold.gaussian(gray_img=gray_img, ksize=2500, offset=-50, object_type="light")

    # save
    Image.fromarray(bin_gauss1).save("tests/test_data/bin_gauss1.png")

    gray_img2 = np.array(Image.open("tests/test_data/z3c1--2022-12-31--09-00-01.png"))[:, :, 1]
    Image.fromarray(gray_img2).save("tests/test_data/gray_img2.png")
    bin_gauss2 = pcv.threshold.gaussian(gray_img=gray_img2, ksize=2500, offset=-50, object_type="light")

    # save
    Image.fromarray(bin_gauss2).save("tests/test_data/bin_gauss2.png")
