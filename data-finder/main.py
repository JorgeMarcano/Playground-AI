import yfinance as yf
import numpy
print("Done importing")

totalLines = 0

def getMinutesFromClose(date):
    #minutes from 4pm -> 16h
    mins = 60 * (16 - date.hour)
    mins -= date.minute

    return mins

#appends all the month into a single data
def getMonth(stock, start_day, startMonth, startMonthDays):
    start_day -= 1
    start_time = "2020-" + str(startMonth) + "-" + str(start_day + 1)
    end_day = (start_day + 7) % startMonthDays
    end_month = startMonth + int((start_day + 7) / startMonthDays)
    end_time = "2020-" + str(end_month) + "-" + str(end_day + 1)
    print(stock, "From:", start_time, "To:", end_time)
    #first week
    wholeData = yf.download(stock, start=start_time, end=end_time, interval="1m")

    #next 3 weeks
    for i in range(1, 4):
        start_day = (start_day + 7) % startMonthDays
        startMonth = startMonth + int((start_day - 7) < 0)
        start_time = "2020-" + str(startMonth) + "-" + str(start_day + 1)
        end_day = (start_day + 7) % startMonthDays
        end_month = startMonth + int((start_day + 7) / startMonthDays)
        end_time = "2020-" + str(end_month) + "-" + str(end_day + 1)

        print(stock, "From:", start_time, "To:", end_time)
        wholeData = wholeData.append(yf.download(stock, start=start_time, end=end_time, interval="1m"))

    #next 2 days
    start_day = (start_day + 7) % startMonthDays
    startMonth = startMonth + int((start_day - 7) < 0)
    start_time = "2020-" + str(startMonth) + "-" + str(start_day + 1)
    end_day = (start_day + 2) % startMonthDays
    end_month = startMonth + int((start_day + 2) / startMonthDays)
    end_time = "2020-" + str(end_month) + "-" + str(end_day + 1)

    print(stock, "From:", start_time, "To:", end_time)
    wholeData = wholeData.append(yf.download(stock, start=start_time, end=end_time, interval="1m"))

    return wholeData

#skips the first day
def parseData(data, file, answerFile):
    global totalLines
    
    totalRows = len(data['Open'])
    dates = data['Open'].index

    currDay = dates[0].day
    dayOpen = data['Open'][0]
    dayClose = data['Close'][0]
    for time in range(1, totalRows):
        #if change of day, save open
        if (dates[time].day != currDay):
            currDay = dates[time].day
            dayOpen = data['Open'][time]
            dayClose = data['Close'][time-1]

        #first 60 minutes, skip (if not 10h30) or first day or if less than 10 minutes from closing
        if ((dates[time].hour > 11 or (dates[time].hour == 10 and dates[time].minute > 31)) and getMinutesFromClose(dates[time]) > 11):
            #get current
            currVal = data['Open'][time]
            #save day open
            string = str((currVal - dayOpen) / currVal) + ", "

            string += str(currVal) + ", "

            #save current and past 10 minutes
            for i in range(1, 11):
                string += str((currVal - data['Open'][time - i]) / currVal) + ", "

            #save 15, 30, 45, 60 min ago
            for i in range(15, 61, 15):
                string += str((currVal - data['Open'][time - i]) / currVal) + ", "

            #save last day close
            string += str((currVal - dayClose) / currVal) + ", "

            #save 1d ago

            #save 1wk ago

            #save hrs till close
            string += str(getMinutesFromClose(dates[time]) / 60) + "\n"#", "

            #save volume
            #string += str(data['Volume'][time]) + "\n"

            #the answer contains 5 minutes and 10 minutes ahead, in %
            if (time+10 >= totalRows):
                continue
            
            answerStr = str((currVal - data['Open'][time + 5]) / currVal) + ", "
            answerStr += str((currVal - data['Open'][time + 10]) / currVal) + "\n"

            file.write(string)
            answerFile.write(answerStr)
            totalLines += 1

#data = yf.download("AAPL", period="7d", interval="1m")

stocks = ["AAPL", "GOOGL", "MCD", "WMT", "IGA", "SBUX", "BBY", "URBN", "HD", "NFLX", "AMZN"]

f = open("../data.txt", "w")
answers = open("../answers.txt", "w")

for i in stocks:
    #save one day before for the close of that day
    print("Getting", "2020-04-21")
    data = yf.download(i, start="2020-04-21", end="2020-04-22")
    data = data.append(getMonth(i, 22, 4, 30))
    print("Parsing Data")
    parseData(data, f, answers)
    print("Done Parsing")

print("A total of", totalLines, "entries have been saved")
f.close()
answers.close()
