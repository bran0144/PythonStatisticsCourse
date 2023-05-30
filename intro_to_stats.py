# Exercises:

# Import numpy with alias np
import numpy as np

# Filter for Belgium
be_consumption = food_consumption[food_consumption['country'] == 'Belgium']

# Filter for USA
usa_consumption = food_consumption[food_consumption['country'] == 'USA']

# Calculate mean and median consumption in Belgium
print(np.mean(be_consumption['consumption']))
print(np.median(be_consumption['consumption']))

# Calculate mean and median consumption in USA
print(np.mean(usa_consumption['consumption']))
print(np.median(usa_consumption['consumption']))

# Subset for Belgium and USA only
be_and_usa = food_consumption[(food_consumption['country'] == "Belgium") | (food_consumption['country'] == 'USA')]

# Group by country, select consumption column, and compute mean and median
print(be_and_usa.groupby('country')['consumption'].agg([np.mean, np.median]))

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Histogram of co2_emission for rice and show plot
rice_consumption['co2_emission'].hist()
plt.show()

# Subset for food_category equals rice
rice_consumption = food_consumption[food_consumption['food_category'] == 'rice']

# Calculate mean and median of co2_emission with .agg()
print(rice_consumption['co2_emission'].agg([np.mean, np.median]))

# Measures of spread
# Variance
dists = msleep['sleep_total'] - np.mean(msleep['sleep_total'])
sq_dist = dists **2
sum_sq_dists = np.sum(sq_dist)
variance = sum_sq_dists / len(msleep) - 1

# Higher variance, means the data is more spread out
# the output is squared

np.var(msleep['sleep_total'], ddof=1)

# Standard deviation
# calculated by taking the square root of the variance
np.sqrt(np.var(msleep['sleep_total'], ddof=1))

np.std(msleep['sleep_total'], ddof=1)

# squares the distances, so longer distances from mean are penalized more than shorter ones

# Mean absolute deviation
np.mean(np.abs(dists))

# penalizes each distance equally

# Quantiles/percentiles
# split up data into equal parts

np.quantile(msleep['sleep_total'], [0, 0,25,0.5,0.75, 1])

# when cut up like this, it represents quartiles which can be visualized using a botplot
plt.boxplot(msleep['sleep_total'])
plt.show()

# can also use linspace as a shortcut
# np.linspace(start, stop, number of quantiles)

np.quantile(msleep['sleep_total'], np.linspace(0,1,5))

# Interquartile Range (IQR)
# distance between 25-75% - the height of the boxplot

np.quantile(msleep['sleep_total', 0.75]) - np.quantile(msleep['sleep_total', 0.25])

from scopy.stats import iqr 
iqr(msleep['sleep_total'])

# Outliers
# standard measure of outliers
# data < Q1 - 1.5 * IQR or  data > Q3 + 1.5 * IQR

iqr = iqr(msleep['bodywt'])
lower_threshold = np.quantile(msleep['bodywt'], 0.25) - 1.5 * iqr
upper_threshold = np.quantile(msleep['bodywt'], 0.75) + 1.5 * iqr

msleep[(msleep['bodywt'] < lower_threshold) | (msleep['bodywt'] > upper_threshold)]

# summary statistics discussed can be shown all at once with
msleep['bodywt'].describe()

# Exercises
# Calculate the quartiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,5)))

# Calculate the quintiles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,6)))

# Calculate the deciles of co2_emission
print(np.quantile(food_consumption['co2_emission'], np.linspace(0,1,11)))

# Print variance and sd of co2_emission for each food_category
print(food_consumption.groupby('food_category')['co2_emission'].agg([np.var, np.std]))

# Import matplotlib.pyplot with alias plt
import matplotlib.pyplot as plt

# Create histogram of co2_emission for food_category 'beef'
food_consumption[food_consumption['food_category'] == 'beef']['co2_emission'].hist()
# Show plot
plt.show()

# Create histogram of co2_emission for food_category 'eggs'
food_consumption[food_consumption['food_category'] == 'eggs']['co2_emission'].hist()
# Show plot
plt.show()

# Calculate total co2_emission per country: emissions_by_country
emissions_by_country = food_consumption.groupby('country')['co2_emission'].sum()

print(emissions_by_country)

# Compute the first and third quantiles and IQR of emissions_by_country
q1 = np.quantile(emissions_by_country, 0.25)
q3 = np.quantile(emissions_by_country, 0.75)
iqr = q3 - q1

# Calculate the lower and upper cutoffs for outliers
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

# Subset emissions_by_country to find outliers
outliers = emissions_by_country[(emissions_by_country < lower) | (emissions_by_country > upper)]
print(outliers)

# Probability
# Exercises

# Count the deals for each product
counts = amir_deals['product'].value_counts()

# Calculate probability of picking a deal with each product
probs = counts / amir_deals['product'].count()
print(probs)

# Set random seed
np.random.seed(24)

# Sample 5 deals without replacement
sample_without_replacement = amir_deals.sample(5)
print(sample_without_replacement)

# Set random seed
np.random.seed(24)

# Sample 5 deals with replacement
sample_with_replacement = amir_deals.sample(5, replace=True)
print(sample_with_replacement)

# Expected value- Mean of a probability distribution
# expected value of a rolling a fair die is 3.5 (1*1/6)+(2*1/6)+(3*1/6)... 

# Probability = area
# P(die roll) <= 2  
# we take the area of each bar representing an outcome of 2 or less
1/6 + 1/6 == 1/3

# Discrete uniform distribution - (all have a discrete value and the same chance of happening)
np.mean(die['number'])

rolls_10 = die.sample(10, replace=True)
rolls_10['number'].hist(bins=np.linspace(1,7,7))
plt.show()

np.mean(rolls_10['number']) == 3.0
mean(die['number']) == 3.5

# law of large numbers
# as sample size increases, the sample mean will approach the theoretical mean

# Create a histogram of restaurant_groups and show plot
restaurant_groups['group_size'].hist(bins=[2,3,4,5,6])
plt.show()

# Create probability distribution
size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0]
# Reset index and rename columns
size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

# Expected value
expected_value = np.sum(size_dist['group_size'] * size_dist['prob'])

# Subset groups of size 4 or more
groups_4_or_more = size_dist[size_dist['group_size'] >= 4]

# Sum the probabilities of groups_4_or_more
prob_4_or_more = np.sum(groups_4_or_more['prob'])
print(prob_4_or_more)

# continuous uniform distribution
# calculate using the area under a horizontal line
# P(4<= wait time <= 7) = 3 * 1/12 = 3/12 (25%)

from scipy.stats import uniform

# calculate the probability of wait time less than 7 (out of 12)
uniform.cdf(7,0,12)

# 0,12 are upper and lower limits
# greater than use 1-
1 - uniform.cdf(7,0,12)

# calculating between 4 and 7
uniform.cdf(7,0,12) - uniform.cdf(4,0,12)

# generating random numbers according to uniform distribution
# (min value, max value, # of random values)
uniform.rvs(0,5,size=10)

# continuous distributions need not be uniform (meaning the probably of some values may be higher than others)
# the area underneath the distribution must still be 1

# Exercises
# Min and max wait times for back-up that happens every 30 min
min_time = 0
max_time = 30

# Import uniform from scipy.stats
from scipy.stats import uniform

# Calculate probability of waiting less than 5 mins
prob_less_than_5 = uniform.cdf(5, min_time, max_time)
print(prob_less_than_5)

# Calculate probability of waiting more than 5 mins
prob_greater_than_5 = 1 - uniform.cdf(5, min_time, max_time)
print(prob_greater_than_5)

# Calculate probability of waiting 10-20 mins
prob_between_10_and_20 = uniform.cdf(20,min_time, max_time) - uniform.cdf(10, min_time, max_time)
print(prob_between_10_and_20)

# Set random seed to 334
np.random.seed(334)

# Import uniform
from scipy.stats import uniform

# Generate 1000 wait times between 0 and 30 mins
wait_times = uniform.rvs(0, 30, size=1000)

# Create a histogram of simulated times and show plot
plt.hist(wait_times)
plt.show()

# Binomial distribution
# outcome with 2 possible values 

from scipy.stats import binom 

# binon.rvs(# of coins, probability of heads/success, size=# of trials)
binom.rvs(1, 0.5, size=8)

# returns an array of random flips (1 or 0 for each flip)
# if we flip 8 coins 1 time:
binom.rvs(8,0.5, size=1)
# we get an array with 1 number for how many 1's (successes) we got

# Binomial distribution
#  - probability distribution of the number of successes in a sequence of independent trials
# outomce here is discrete
#  n = number of trials
#  p = probability of success

# probability of 7 heads in 10 trials
# binon.pmf(# of heads, # of trials, probability of heads)
binom.pmf(7,10,0.5)

# probability of getting 7 or fewer heads
binom.cdf(7, 10, 0.5)

# expected value = calculated as n * p
# if trials aren't indpendent, we can't use binomial distribution

# Exercises:
# Import binom from scipy.stats
from scipy.stats import binom

# Set random seed to 10
np.random.seed(10)

# Simulate a single deal
print(binom.rvs(1, 0.3, size=1))

# Simulate 1 week of 3 deals
print(binom.rvs(3, 0.3, size=1))

# Simulate 52 weeks of 3 deals
deals = binom.rvs(3, 0.3, size=52)

# Print mean deals won per week
print(np.mean(deals))

# Probability of closing 3 out of 3 deals
prob_3 = binom.pmf(3, 3, 0.3)

print(prob_3)

# Probability of closing <= 1 deal out of 3 deals
prob_less_than_or_equal_1 = binom.cdf(1,3, 0.3)

print(prob_less_than_or_equal_1)

# Probability of closing > 1 deal out of 3 deals
prob_greater_than_1 = 1- binom.cdf(1,3,0.3)

print(prob_greater_than_1)

# Expected number won with 30% win rate
won_30pct = 3 * 0.3
print(won_30pct)

# Expected number won with 25% win rate
won_25pct = 3 * 0.25
print(won_25pct)

# Expected number won with 35% win rate
won_35pct = 3 * 0.35
print(won_35pct)

# Normal distribution
# symmetrical
# area underneath =1
# probability never hits 0
# described by its mean and standard deviation
# when it has a mean of 0 and std of 1, its called a standard normal distribution
# 68% of the area is within 1 std of the mean
# 95% falls within 2 std of the mean
# 99.7% falls within 3 std of the mean

from scipy.stats import norm 
# what % of women are shorter than 154?
norm.cdf(154, 161, 7)
# (# we want, mean, std)
# of women between 154-157 cm
norm.cdf(157, 161, 7) - norm.cdf(154, 161, 7)

# what height 90% of women are shorter than?
norm.ppf(0.9, 161, 7)

# what height are 90% of women taller than?
norm.ppf((1-0.9), 161,7)

# generating random numbers with normal dist
norm.rvs(161,7,size=10)

# Histogram of amount with 10 bins and show plot
amir_deals['amount'].hist(bins=10)
plt.show()

# Probability of deal < 7500
prob_less_7500 = norm.cdf(7500, 5000, 2000)

print(prob_less_7500)

# Probability of deal > 1000
prob_over_1000 = 1- norm.cdf(1000, 5000, 2000)

print(prob_over_1000)

# Probability of deal between 3000 and 7000
prob_3000_to_7000 = norm.cdf(7000, 5000, 2000) - norm.cdf(3000, 5000, 2000)

print(prob_3000_to_7000)

# Calculate amount that 25% of deals will be less than
pct_25 = norm.ppf(.25, 5000, 2000)

print(pct_25)

# Calculate new average amount
new_mean = 5000 * 1.2

# Calculate new standard deviation
new_sd = 2000 * 1.3

# Simulate 36 new sales
new_sales = norm.rvs(new_mean, new_sd, size=36)

# Create histogram and show
plt.hist(new_sales)
plt.show()

# Central Limit Theore
