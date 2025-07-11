import numpy as np


def get_modified_action(ppfd: float, channel: int, offset: float) -> np.ndarray:
    adjustment = -offset / (((offset + BALANCED_ACTION[channel]) / ppfd) - 1)

    modified_action = BALANCED_ACTION.copy()
    modified_action[channel] += adjustment
    modified_action = modified_action / (ppfd + adjustment) * ppfd

    return modified_action


OLD_BALANCED_ACTION = np.array([22.5, 81.0, 9.3, 0.0, 7.2, 14.2])
BALANCED_ACTION = OLD_BALANCED_ACTION / np.sum(OLD_BALANCED_ACTION[:5]) * 105.0

DIM_ACTION = 0.675 * BALANCED_ACTION

RED_ACTION = get_modified_action(ppfd=105.0, channel=4, offset=50.0)
BLUE_ACTION = get_modified_action(ppfd=105.0, channel=0, offset=50.0)

TWILIGHT_INTENSITIES_30_MIN = np.array(
    [
        0.002827982971,
        0.01109596503,
        0.02471504108,
        0.0435358277,
        0.06735132485,
        0.09590206704,
        0.1288751694,
        0.1659110052,
        0.2066022518,
        0.2505043837,
        0.2971366267,
        0.3459895889,
        0.3965281229,
        0.4481989566,
        0.5004364171,
        0.5526691081,
        0.604325633,
        0.654840319,
        0.7036608481,
        0.7502520726,
        0.7941046008,
        0.8347386124,
        0.8717086279,
        0.9046111404,
        0.9330855692,
        0.9568199834,
        0.9755549175,
        0.9890862332,
        0.9972641654,
    ]
)
