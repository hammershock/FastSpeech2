from text.symbols import symbols

path1 = "./train.txt"

phones = []
with open(path1, "r") as f:
    for line in f:
        parts = line.split("|")
        phones += parts[2][1:-1].split()

print(symbols)
s = set(phones)
outliers = {item for item in s if "@" + item not in symbols}
print(outliers)

path1 = "./val.txt"

phones = []
with open(path1, "r") as f:
    for line in f:
        parts = line.split("|")
        phones += parts[2][1:-1].split()

s = set(phones)
outliers = {item for item in s if "@" + item not in symbols}
print(outliers)