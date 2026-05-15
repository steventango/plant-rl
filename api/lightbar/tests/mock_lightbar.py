from api.lightbar.app.lightbar import Lightbar


class MockSMBus:
    def __init__(self):
        self.data = {}
        self.reset_calls = 0

    def write_i2c_block_data(self, address, register, data):
        if address not in self.data:
            self.data[address] = {}
        if register not in self.data[address]:
            self.data[address][register] = []
        self.data[address][register].append(data)

    def write_byte(self, address, value):
        if address == 0x00 and value == 0x06:
            self.reset_calls += 1


class MockLightbar(Lightbar):
    def get_i2c(self):
        return MockSMBus()
