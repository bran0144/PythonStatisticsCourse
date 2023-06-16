# Intro to regression

# Linear regression - response variable is numeric
# and logistic regression - reponse varialbe is logical (yes or no)
# simple - means you only have one explanatory variable
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.scatterplot(x="n_claims", y="total_payment_sek", data=swedish_motor_insurance)
plt.show()

sns.regplot(x="n_claims", y="total_payment_sek", data=swedish_motor_insurance, ci=None)
plt.show()

# Exercises:
import matplotlib.pyplot as plt 
import seaborn as sns 
# Draw the scatter plot
sns.scatterplot(x="n_convenience", y="price_twd_msq", data=taiwan_real_estate)

# Show the plot
plt.show()


# Draw a trend line on the scatter plot of price_twd_msq vs. n_convenience
sns.regplot(x="n_convenience",
         y="price_twd_msq",
         data=taiwan_real_estate,
         ci=None,
         scatter_kws={'alpha': 0.5})

# Show the plot
plt.show()

# FItting a line

# slope, y intercept

from statsmodel.formula.api import ols

mdl_payment_vs_claims = ols("total_payment_sek ~ n_claims", data=swedish_motor_insurance)
mdl_payment_vs_claims = mdl_payment_vs_claims.fit()
print(mdl_payment_vs_claims.params)

# returns 2 coefficients, the slope and the y intercept

# Exercises
# Import the ols function
from statsmodels.formula.api import ols

# Create the model object
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience", data=taiwan_real_estate)

# Fit the model
mdl_price_vs_conv = mdl_price_vs_conv.fit()

# Print the parameters of the fitted model
print(mdl_price_vs_conv.params)

# Categorical explanatory variables

# should draw a histogram for each of the species
sns.displot(data=fish, x="mass_g", col="species", col_wrap=2, bins=9)
plt.show()

# displot gives a separate panel for each species
# by default, displot creates histograms

summary_stats = fish.groupby("species")["mass_g"].mean()
print(summary_stats)

mdl_mass_vs_speciies = ols("mass_g ~ species", data=fish).fit()
print(mdl_mass_vs_speciies.params)

mdl_mass_vs_speciies = ols("mass_g ~ species + 0", data=fish).fit()
print(mdl_mass_vs_speciies.params)

# When you have a single categofical explanatory variable, the linear regression
# coefficients are simply the means of each category

# Exercises
# Histograms of price_twd_msq with 10 bins, split by the age of each house
sns.displot(data=taiwan_real_estate,
         x="price_twd_msq",
         col="house_age_years",
         bins=10)

# Show the plot
plt.show()

# Calculate the mean of price_twd_msq, grouped by house age
mean_price_by_age = taiwan_real_estate.groupby("house_age_years")["price_twd_msq"].mean()

# Print the result
print(mean_price_by_age)

# Create the model, fit it
mdl_price_vs_age = ols("price_twd_msq ~ house_age_years", data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age.params)

# Update the model formula to remove the intercept
mdl_price_vs_age0 = ols("price_twd_msq ~ house_age_years + 0", data=taiwan_real_estate).fit()

# Print the parameters of the fitted model
print(mdl_price_vs_age0.params)

# Making predictions



