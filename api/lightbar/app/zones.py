class Zone:
    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right


ZONES = {
    "alliance-zone1": Zone(0x25, 0x24),
    "alliance-zone2": Zone(0x24, 0x26),
    "alliance-zone3": Zone(0x41, 0x42),
    "alliance-zone4": Zone(0x49, 0x54),
    "alliance-zone5": Zone(0x38, 0x40),
    "alliance-zone6": Zone(0x12, 0x13),
    "alliance-zone7": Zone(0x48, 0x51),
    "alliance-zone8": Zone(0x18, 0x19),
    "alliance-zone9": Zone(0x14, 0x15),
    "alliance-zone10": Zone(0x30, 0x39),
    "alliance-zone11": Zone(0x34, 0x35),
    "alliance-zone12": Zone(0x08, 0x09),
    "mitacs-zone1": Zone(0x25, 0x33),
    "mitacs-zone2": Zone(0x69, 0x71),
    "mitacs-zone3": Zone(0x1B, 0x1C),
    "mitacs-zone6": Zone(0x58, 0x67),
    "mitacs-zone8": Zone(0x10, 0x11),
    "mitacs-zone9": Zone(0x63, 0x65),
}
