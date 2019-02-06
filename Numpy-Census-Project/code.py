# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data = np.genfromtxt(path, delimiter=",", skip_header=1)
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
census = np.concatenate([data,new_record])
#Code starts here



# --------------
#Code starts here
age = census[:, 0]
max_age = np.max(age)
min_age = np.min(age)
age_mean = np.mean(age)
age_std = np.std(age)
#age_std
#max_age = np.max(age)
#max_age


# --------------
#Code starts here
race_0 = census[census[:, 2] == 0]
race_1 = census[census[:, 2] == 1]
race_2 = census[census[:, 2] == 2]
race_3 = census[census[:, 2] == 3]
race_4 = census[census[:, 2] == 4]
len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)

minority_race = 3
#len_2 = np.size(race_2)
#len_3 = np.size(race_3)
#len_4 = np.size(race_4)



# --------------
#Code starts here
senior_citizens = census[census[:, 0] > 60]
working_hours_sum = np.sum(census[census[:,0] > 60, 6])
senior_citizens_len = len(senior_citizens)
avg_working_hours = (working_hours_sum)/(senior_citizens_len)
print(avg_working_hours)


# --------------
#Code starts here
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
#avg_pay_high = np.mean(census[census[:,1]>10],7)
#avg_pay_low = np.mean(census[census[:,1]<=10],7)
avg_pay_high = np.mean(census[census[:,1] > 10, 7])
avg_pay_low = np.mean(census[census[:,1] <= 10, 7])
(avg_pay_high == avg_pay_low).all()


