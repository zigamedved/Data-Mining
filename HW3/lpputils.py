import datetime

FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def parsedate(x):
    if not isinstance(x, datetime.datetime):
        x = datetime.datetime.strptime(x, FORMAT)
    return x

def tsdiff(x, y):
    return (parsedate(x) - parsedate(y)).total_seconds()

def tsadd(x, seconds):
    d = datetime.timedelta(seconds=seconds)
    nd = parsedate(x) + d
    return nd.strftime(FORMAT)

if __name__ == "__main__":
    testd1 = "2012-01-01 23:32:38.000"
    testd2 = "2012-12-01 03:33:38.000"
    
    testd1 = datetime.datetime.strptime(testd1, FORMAT)

    for i in range(23000):
        a = tsdiff(testd1, testd2)
        b = tsadd(testd1, -122)
