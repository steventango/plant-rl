import json
from datetime import date, datetime, time
from pathlib import Path


class TimeFrame:
    """
    Represents a range of time
    """

    def __init__(self, date: date, time1: time, time2: time) -> None:
        assert time1 < time2

        self.datetime1: datetime = datetime.combine(date, time1)
        self.datetime2: datetime = datetime.combine(date, time2)

    def time_in_frame(self, daytime_to_check: datetime) -> bool:
        # print(f"Is {self.datetime1} < {daytime_to_check} < {self.datetime2}?")
        if self.datetime1 < daytime_to_check < self.datetime2:
            # print("yes")
            return True
        # print("no")
        return False

    def __str__(self) -> str:
        return f"TimeFrame: {str(self.datetime1)} --> {str(self.datetime2)}"

    def get_date(self):
        return str(self.datetime1.date())

    def get_time1(self):
        return str(self.datetime1.time()).removesuffix(":00")

    def get_time2(self):
        return str(self.datetime2.time()).removesuffix(":00")


class TimeFrameList:
    def __init__(self) -> None:
        self.data: list = []

    def load_from_file(self, path: Path) -> bool:
        try:
            with open(path, "r") as f:
                l = json.load(f)
                for tfd in l:
                    # {'date': '2022-04-12', 'ontime': '12:00', 'offtime': '13:00'}
                    day: date = datetime.strptime(tfd["date"], "%Y-%m-%d").date()
                    time1: time = datetime.strptime(tfd["ontime"], "%H:%M").time()
                    time2: time = datetime.strptime(tfd["offtime"], "%H:%M").time()
                    self.data.append(TimeFrame(day, time1, time2))
        except Exception as e:
            print("ERROR", e)
            return False
        return True

    def load_from_list(self, l: list) -> bool:
        try:
            for tfd in l:
                # {'date': '2022-04-12', 'ontime': '12:00', 'offtime': '13:00'}
                day: date = datetime.strptime(tfd["date"], "%Y-%m-%d").date()
                time1: time = datetime.strptime(tfd["ontime"], "%H:%M").time()
                time2: time = datetime.strptime(tfd["offtime"], "%H:%M").time()
                self.data.append(TimeFrame(day, time1, time2))
        except Exception as e:
            print("ERROR", e)
            return False
        return True

    def time_in_list(self, daytime: datetime) -> bool:
        for tf in self.data:
            tf: TimeFrame
            if tf.time_in_frame(daytime):
                return True
        return False

    def save_to_file(self, dirpath: Path):
        json_l = []
        for tf in self.data:
            tf: TimeFrame
            json_l.append(
                {
                    "date": tf.get_date(),
                    "ontime": tf.get_time1(),
                    "offtime": tf.get_time2(),
                }
            )

        with open(dirpath / "timeframe.json", "w") as f:
            json.dump(json_l, f)
