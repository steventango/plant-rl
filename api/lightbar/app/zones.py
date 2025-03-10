class Zone:
    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right


ZONES = {
    1: Zone(0x25, 0x33),
    2: Zone(0x69, 0x71),
    3: Zone(0x1b, 0x1c),
    6: Zone(0x58, 0x67),
    8: Zone(0x10, 0x11),
    9: Zone(0x63, 0x65)
}
