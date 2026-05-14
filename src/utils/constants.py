import numpy as np


def get_modified_action(ppfd: float, channel: int, offset: float) -> np.ndarray:
    adjustment = -offset / (((offset + BALANCED_ACTION_105[channel]) / ppfd) - 1)

    modified_action = BALANCED_ACTION_105.copy()
    modified_action[channel] += adjustment
    modified_action = modified_action / (ppfd + adjustment) * ppfd

    return modified_action


def adjust_ppfd(action: np.ndarray, ppfd: float) -> np.ndarray:
    """
    Adjust the action to match the given PPFD.
    """
    adjusted_action = action.copy()
    adjusted_action /= np.sum(adjusted_action[:5])
    adjusted_action *= ppfd
    return adjusted_action


BALANCED_ACTION_105 = np.array([19.5, 71.53, 7.82, 0.0, 6.15, 14.12])
BALANCED_ACTION_120 = adjust_ppfd(BALANCED_ACTION_105, 120.0)
BALANCED_ACTION_100 = adjust_ppfd(BALANCED_ACTION_105, 100.0)

DIM_ACTION = 0.675 * BALANCED_ACTION_100.copy()

RED_ACTION = get_modified_action(ppfd=105.0, channel=4, offset=40.0)
BLUE_ACTION = get_modified_action(ppfd=105.0, channel=0, offset=40.0)
