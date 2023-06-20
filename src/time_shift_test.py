import datetime
import time

from Prior import Prior

datetime_0 = datetime.datetime(1970,1,1,0,0,0)  # Unix epoch

# High tide experiment week
high_tide_1_2023_06_20 = datetime.datetime(2023,6,20,1,40)
high_tide_2_2023_06_20 = datetime.datetime(2023,6,20,14,10)
high_tide_1_2023_06_21 = datetime.datetime(2023,6,21,2,30)
high_tide_2_2023_06_21 = datetime.datetime(2023,6,21,15,0)
high_tide_1_2023_06_22 = datetime.datetime(2023,6,22,2,50)
high_tide_2_2023_06_22 = datetime.datetime(2023,6,22,15,40)
high_tide_1_2023_06_23 = datetime.datetime(2023,6,23,3,10)
high_tide_2_2023_06_23 = datetime.datetime(2023,6,23,16,10)
high_tide_1_2023_06_24 = datetime.datetime(2023,6,24,4,10)
high_tide_2_2023_06_24 = datetime.datetime(2023,6,24,16,50)

# High tide prior
high_tide_1_2022_06_21 = datetime.datetime(2022,6,21,5,30)
high_tide_2_2022_06_21 = datetime.datetime(2022,6,21,18,30)
high_tide_1_2022_06_22 = datetime.datetime(2022,6,22,6,50)
high_tide_2_2022_06_22 = datetime.datetime(2022,6,22,19,40)


prior_date = datetime.datetime(2022,6,21)
today_date = datetime.datetime(2023,6,20)
high_tide_prior = high_tide_1_2022_06_21
high_tide_today = high_tide_1_2023_06_20

# 
print("This is the prior date: ", prior_date)
print("This is the today date: ", today_date)
print("This is the high tide prior: ", high_tide_prior)
print("This is the high tide today: ", high_tide_today)
print("This is the high tide prior in seconds: ", high_tide_prior.timestamp())
print("This is the high tide today in seconds: ", high_tide_today.timestamp())

print("This is the time now: ", datetime.datetime.now())
print("This is the time now in seconds: ", datetime.datetime.now().timestamp())
print("This is the time now in time.time(): ", time.time())
print("Diff in seconds between now and time.time(): ", datetime.datetime.now().timestamp() - time.time())


# Diff date 
diff = today_date - prior_date
print("This is the difference between prior and now in datetime: ", diff)
diff_seconds = diff.total_seconds()
total_diff_seconds = int((high_tide_today - high_tide_prior).total_seconds())
print("This is the difference between prior and now in seconds: ", diff_seconds)
diff_tide_s = int((high_tide_today - high_tide_prior).total_seconds()) - diff_seconds
print("Diff prior tide and now in seconds: ", diff_tide_s)
print("Diff prior tide and now in datetime: ", datetime.timedelta(seconds=diff_tide_s))
print("Diff prior tide and now in hours: ", diff_tide_s/3600)

print("Diff now and high tide today in seconds: ", int((high_tide_today - datetime.datetime.now()).total_seconds()))
print("Diff now and high tide today in hours: ", (high_tide_today - datetime.datetime.now()).total_seconds()/3600)
print("Diff now and high tide prior in seconds: ", int((high_tide_prior - datetime.datetime.now()).total_seconds()))
print("Diff now and high tide prior in hours: ", (high_tide_prior - datetime.datetime.now()).total_seconds()/3600)


print("Time <now> in prior time: ", datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp() - total_diff_seconds))


