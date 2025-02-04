from api.lightbar.app.lightbar import Lightbar


class MockSMBus:
    def __init__(self):
        self.data = {}

    def write_i2c_block_data(self, address, register, data):
        if address not in self.data:
            self.data[address] = {}
        if register not in self.data[address]:
            self.data[address][register] = []
        self.data[address][register].append(data)


class MockLightbar(Lightbar):
    def get_i2c(self):
        return MockSMBus()
