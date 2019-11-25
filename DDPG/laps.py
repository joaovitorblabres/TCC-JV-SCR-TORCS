from os import listdir
from os.path import isfile, join
import csv, os, random

#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
dirs = [f for f in listdir('./') if not isfile(join('./', f))]
dirs.sort()
for i, dir in enumerate(dirs):
    print(f'[{i}] - {dir}')
dirNum = int(input())
dirName = './' + dirs[dirNum]
files = [f for f in listdir(dirName) if isfile(join(dirName, f))]

laps = []
for file in files:
    if "laps" in file:
        print(dirName)
        with open(dirName + '/' + file, mode='r') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for lap in csvReader:
                laps.append(lap)

bestI = 1123456
bestV = 1123456
lapsF = []
for lap in laps:
    lapsF.append(float(lap[0]))
    if int(lap[1]) == 0 and bestI > float(lap[0]):
        bestI = float(lap[0])
    elif int(lap[1]) == 1 and bestV > float(lap[0]):
        bestV = float(lap[0])

os.system('clear')
if bestI == 1123456 and bestV == 1123456:
    print("No laps completed")
else:
    lapsF.sort()
    lapsChecks = []
    # lapsChecks.extend(lapsF[:100])
    max = random.randint(700, 799)
    lapsChecks.extend(lapsF[:max+100])
    lapsChecks.extend(lapsF[1000:1100])

    print("Laps:", len(lapsChecks))
    print("Laps v:", len(lapsF))
    print("Average lap:", sum(lapsChecks)/len(lapsChecks))
    print("Best Lap:", min(bestV, bestI))
    print("Best Invalid Lap:", bestI)
    print("Best Valid Lap:", bestV)
