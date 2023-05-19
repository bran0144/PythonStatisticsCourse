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


