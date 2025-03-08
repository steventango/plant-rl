class Zone:
    def __init__(self, left: int, right: int):
        self.left = left
        self.right = right


ZONES = {
    2: Zone(0x69, 0x71),
    8: Zone(0x10, 0x11),
    9: Zone(0x63,0x65)
}
