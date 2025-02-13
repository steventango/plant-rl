from abc import ABC, abstractmethod
import re
from datetime import datetime


class Regex(ABC):
    def __init__(self, pattern: str, name: str):
        self.pattern = pattern
        self.name = name

    @abstractmethod
    def getdate(self, input_string: str) -> datetime:
        pass


class DateRegex1(Regex):
    def __init__(self):
        super().__init__(r"z\d*c\d*--(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})", "Date1")
        # z2c1--2022-04-14--12-00-01.png

    def getdate(self, input_string: str):
        # Implement how to extract date using this specific regex pattern
        res = re.findall(self.pattern, input_string)
        # print(f"input: {input_string} res: {res}")
        if len(res) != 1:
            return datetime(1,1,1,1,1,1)
        year = int(res[0][0])
        month = int(res[0][1])
        day = int(res[0][2])
        hour = int(res[0][3])
        minute = int(res[0][4])
        second = int(res[0][5])
        return datetime(year,month,day,hour,minute,second)

class DateRegex2(Regex):
    def __init__(self):
        super().__init__(r"zone\d{2}cam\d{2}-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", "Date1")
        # zone01cam02-2024-02-05-17-40-01.png

    def getdate(self, input_string: str):
        # Implement how to extract date using this specific regex pattern
        res = re.findall(self.pattern, input_string)
        # print(f"input: {input_string} res: {res}")
        if len(res) != 1:
            return datetime(1,1,1,1,1,1)
        year = int(res[0][0])
        month = int(res[0][1])
        day = int(res[0][2])
        hour = int(res[0][3])
        minute = int(res[0][4])
        second = int(res[0][5])
        return datetime(year,month,day,hour,minute,second)


class RegexBuilder:
    def __init__(self, selection: str) -> None:
        self.selection = selection

    def get_regex(self) -> Regex:
        if self.selection == "Date1":
            r = DateRegex1()
        elif self.selection == "Date2":
            r = DateRegex2()
        return r


# Example usage:
# date_regex1 = DateRegex1()
# date_regex2 = DateRegex2()

# input_string = "Date: 2024-02-08"
# date1 = date_regex1.getdate(input_string)
# print("Date 1:", date1)

# input_string = "Date: 02/08/2024"
# date2 = date_regex2.getdate(input_string)
# print("Date 2:", date2)
