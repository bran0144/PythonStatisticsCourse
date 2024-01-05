# Parallel slopes linear regression

# when using one explanatory variable at a time
from statsmodels.formula.api import ols 
mdl_mass_vs_length = ols("mass_g ~ length_cm", data=fish).fit()
# response variable on left, explanatory variable on right

# using categorical variable
mdl_mass_vs_species = ols("mass_g ~ species + 0", data=fish).fit()

# Both variables at the same time
mdl_mass_vs_both = ols("mass_g ~ length_cm + species + 0", data=fish).fit()

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.reglot(x="length_cm", y="mass_g", data=fish, ci=None)
plt.show()

sns.boxplot(x="species", y="mass_g", data=fish, showmeans=True)

coeffs = mdl_mass_vs_both.params 
ic_bream, ic_perch, ic_pike, ic_roach, sl = coeffs

sns.scatterplot(x="length_cm", y="mass_g", hue="species", data=fish)

plt.axline(xy1=(0, ic_bream), slope=sl, color="blue")
plt.axline(xy1=(0, ic_perch), slope=sl, color="green")
plt.axline(xy1=(0, ic_pike), slope=sl, color="red")
plt.axline(xy1=(0, ic_roach), slope=sl, color="orange")

# Exercises

# Import ols from statsmodels.formula.api
from statsmodels.formula.api import ols

# Fit a linear regression of price_twd_msq vs. n_convenience
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience", data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_conv.params)

# Import ols from statsmodels.formula.api
from statsmodels.formula.api import ols

# Fit a linear regression of price_twd_msq vs. n_convenience
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience",
                        data=taiwan_real_estate).fit()

# Fit a linear regression of price_twd_msq vs. house_age_years, no intercept
mdl_price_vs_age = ols("price_twd_msq ~ house_age_years + 0",
                        data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_age.params)

# Import ols from statsmodels.formula.api
from statsmodels.formula.api import ols

# Fit a linear regression of price_twd_msq vs. n_convenience
mdl_price_vs_conv = ols("price_twd_msq ~ n_convenience",
                        data=taiwan_real_estate).fit()

# Fit a linear regression of price_twd_msq vs. house_age_years, no intercept
mdl_price_vs_age = ols("price_twd_msq ~ house_age_years + 0", data=taiwan_real_estate).fit()

# Fit a linear regression of price_twd_msq vs. n_convenience plus house_age_years, no intercept
mdl_price_vs_both = ols("price_twd_msq ~ n_convenience + house_age_years + 0", data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both.params)

# Import matplotlib.pyplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot with linear trend line of price_twd_msq vs. n_convenience
sns.regplot(x="n_convenience", y="price_twd_msq", data=taiwan_real_estate, ci=None)

# Show the plot
plt.show()

# Import matplotlib.pyplot and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a boxplot of price_twd_msq vs. house_age_years
sns.boxplot(x="house_age_years", y="price_twd_msq", data=taiwan_real_estate, showmeans=True)

# Show the plot
plt.show()

# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Print coeffs
print(coeffs)

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Draw a scatter plot of price_twd_msq vs. n_convenience colored by house_age_years
sns.scatterplot(x="n_convenience", y="price_twd_msq", hue="house_age_years", data=taiwan_real_estate)

# Show the plot
plt.show()

# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Draw a scatter plot of price_twd_msq vs. n_convenience, colored by house_age_years
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                hue="house_age_years",
                data=taiwan_real_estate)

# Add three parallel lines for each category of house_age_years
# Color the line for ic_0_15 blue
plt.axline(xy1=(0, ic_0_15), slope=slope, color="blue")
# Color the line for ic_15_30 orange
plt.axline(xy1=(0, ic_15_30), slope=slope, color="orange")
# Color the line for ic_30_45 green
plt.axline(xy1=(0, ic_30_45), slope=slope, color="green")

# Show the plot
plt.show()

# Predicting parallel slopes
import pandas as pd 
import numpy as np 
expl_data_length = pd.DatatFrame({"length_cm": np.arange(5, 61, 4)})
print(expl_data_length)

# to make a df for multiple explanatory variables
from itertools import product 
product(["A", "B", "C"], [1,2])

# returns Cartesian product of your input variables

length_cm  = np.arange(5, 61, 5)
species = fish['species'].unique()
p = product(length_cm, species)

expl_data_both = pd.DataFrame(p, columns=['length_cm', 'species'])
predict_data_length = expl_data_length.assign(mass_g = mdl_mass_vs_length.predict(expl_data))

predict_data_both = expl_data_both.assign(mass_g = mdl_mass_vs_both.predict(expl_data))

plt.axline(xy1=(0, ic_bream), slope=sl, color="blue")
plt.axline(xy1=(0, ic_perch), slope=sl, color="green")
plt.axline(xy1=(0, ic_pike), slope=sl, color="red")
plt.axline(xy1=(0, ic_roach), slope=sl, color="orange")
sns.scatterplot(x="length_cm", y="mass_g", hue="species", data=fish)

sns.scatterplot(x="length_cm", y="mass_g", color="black", data=prediction_data)

coeffs = mdl_mass_vs_length.params 
intercept, slope = coeffs 

explanatory_data = pd.DataFrame({'length_cm': np.arange(5, 61, 5)})

prediction_data = explanatory_data.assign(mass_g = intercept + slope * explanatory_data)

coeffs = mdl_mass_vs_both.params 
ic_bream, ic_perch, ic_pike, ic_roach, sl = coeffs

conditions = [
    explanatory_data['species'] == 'Bream',
    explanatory_data['species'] == 'Perch',
    explanatory_data['species'] == 'Pike',
    explanatory_data['species'] == 'Roach',
]
choices = [ic_bream, ic_perch, ic_pike, ic_roach]
intercept = np.select(conditions, choices)

prediction_data = explanatory_data.assign(
    intercept = np.select(conditions, choices),
    mass_g = intercept + slope * explanatory_data['length_cm'])

mdl_mass_vs_both.predict(explanatory_data)

# Exercises

# Create n_convenience as an array of numbers from 0 to 10
n_convenience = [0,1,2,3,4,5,6,7,8,9,10]

# Extract the unique values of house_age_years
house_age_years = taiwan_real_estate['house_age_years'].unique()

# Create p as all combinations of values of n_convenience and house_age_years
p = product(n_convenience, house_age_years)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(p, columns=['n_convenience', 'house_age_years'])

print(explanatory_data)

# Add predictions to the DataFrame
prediction_data = explanatory_data.assign(price_twd_msq = mdl_price_vs_both.predict(explanatory_data))

print(prediction_data)

# Extract the model coefficients, coeffs
coeffs = mdl_price_vs_both.params

# Assign each of the coeffs
ic_0_15, ic_15_30, ic_30_45, slope = coeffs

# Create the parallel slopes plot
plt.axline(xy1=(0, ic_0_15), slope=slope, color="green")
plt.axline(xy1=(0, ic_15_30), slope=slope, color="orange")
plt.axline(xy1=(0, ic_30_45), slope=slope, color="blue")
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                hue="house_age_years",
                data=taiwan_real_estate)

# Add the predictions in black
sns.scatterplot(x="n_convenience", y="price_twd_msq", color="black", data=prediction_data)

plt.show()

# Define conditions
conditions = [
    explanatory_data["house_age_years"] == "0 to 15",
    explanatory_data["house_age_years"] == "15 to 30",
    explanatory_data["house_age_years"] == "30 to 45"]

# Define choices
choices = [ic_0_15, ic_15_30, ic_30_45]

# Create array of intercepts for each house_age_year category
intercept = np.select(conditions, choices)

# Create prediction_data with columns intercept and price_twd_msq
prediction_data = explanatory_data.assign(
    intercept = intercept,
    price_twd_msq = intercept + slope * explanatory_data["n_convenience"])

print(prediction_data)

# Assessing model performance

# coefficient of determination (R squared) - how well line fits
# residual standard error (RSE) - typical size of residuals

print(mdl_mass_vs_length.rsquared)
print(mdl_mass_vs_species.rsquared)
print(mdl_mass_vs_both.rsquared)

# overfitting - good for the particular dataset, but does not reflect general population
# caused by too many explantory variables
# adjusted coefficient of determination adds a penalty with more explanatory variables
# rsquared_adj

print(mdl_mass_vs_both.rsquared_adj)

rse_length = np.sqrt(mdl_mass_vs_length.mse_resid)

# Exercises

# Print the coeffs of determination for mdl_price_vs_conv
print("rsquared_conv: ", mdl_price_vs_conv.rsquared)
print("rsquared_adj_conv: ", mdl_price_vs_conv.rsquared_adj)

# Print the coeffs of determination for mdl_price_vs_age
print("rsquared_age: ", mdl_price_vs_age.rsquared)
print("rsquared_adj_age: ", mdl_price_vs_age.rsquared_adj)

# Print the coeffs of determination for mdl_price_vs_both
print("rsquared_both: ", mdl_price_vs_both.rsquared)
print("rsquared_adj_both: ", mdl_price_vs_both.rsquared_adj)

# Print the RSE for mdl_price_vs_conv
print("rse_conv: ", np.sqrt(mdl_price_vs_conv.mse_resid))

# Print the RSE for mdl_price_vs_conv
print("rse_age: ", np.sqrt(mdl_price_vs_age.mse_resid))

# Print RSE for mdl_price_vs_conv
print("rse_both: ", np.sqrt(mdl_price_vs_both.mse_resid))

bream = fish[fish["species"] == "Bream"]
perch = fish[fish["species"] == "Perch"]
pike = fish[fish["species"] == "Pike"]
roach = fish[fish["species"] == "Roach"]

mdl_bream = ols("mass_g ~ length_cm", data=bream).fit()
print(mdl_bream.params)
mdl_perch = ols("mass_g ~ length_cm", data=perch).fit()
mdl_pike = ols("mass_g ~ length_cm", data=pike).fit()
mdl_roach = ols("mass_g ~ length_cm", data=roach).fit()

explanatory_data = pd.DataFrame({"length_cm": np.arange(5, 61, 5)})

prediction_data_bream = explanatory_data.assign(mass_g = mdl_bream.predict(explanatory_data),
    species="Bream")
# do the same for all the other species

prediction_data = pd.concat(prediction_data_bream, prediction_data_roach, prediction_data_perch, prediction_data_pike)

sns.lmplot(x="length_cm", y="mass_g", data=fish, hue="species", ci=None)
sns.scatterplot(x="length_cm", y="mass_g",data=prediction_data, hue="species", ci=None, legend=False)

plt.show()

mdl_fish = ols("mass_g ~ length_cm", data=fish).fit()
print(mdl_fish.rsquared_adj)

# Filter for rows where house age is 0 to 15 years
taiwan_0_to_15 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "0 to 15"]

# Filter for rows where house age is 15 to 30 years
taiwan_15_to_30 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "15 to 30"]

# Filter for rows where house age is 30 to 45 years
taiwan_30_to_45 = taiwan_real_estate[taiwan_real_estate["house_age_years"] == "30 to 45"]

# Model price vs. no. convenience stores using 0 to 15 data
mdl_0_to_15 = ols("price_twd_msq ~ n_convenience", data=taiwan_0_to_15).fit()

# Model price vs. no. convenience stores using 15 to 30 data
mdl_15_to_30 = ols("price_twd_msq ~ n_convenience", data=taiwan_15_to_30).fit()

# Model price vs. no. convenience stores using 30 to 45 data
mdl_30_to_45 = ols("price_twd_msq ~ n_convenience", data=taiwan_30_to_45).fit()

# Print the coefficients
print(mdl_0_to_15.params)
print(mdl_15_to_30.params)
print(mdl_30_to_45.params)

# Create explanatory_data, setting no. of conv stores from  0 to 10
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0, 11)})

# Add column of predictions using "0 to 15" model and explanatory data 
prediction_data_0_to_15 = explanatory_data.assign(price_twd_msq = mdl_0_to_15.predict(explanatory_data))

# Same again, with "15 to 30"
prediction_data_15_to_30 = explanatory_data.assign(price_twd_msq = mdl_15_to_30.predict(explanatory_data))

# Same again, with "30 to 45"
prediction_data_30_to_45 = explanatory_data.assign(price_twd_msq = mdl_30_to_45.predict(explanatory_data))

print(prediction_data_0_to_15)
print(prediction_data_15_to_30)
print(prediction_data_30_to_45)

# Plot the trend lines of price_twd_msq vs. n_convenience for each house age category
sns.lmplot(x="n_convenience",
           y="price_twd_msq",
           data=taiwan_real_estate,
           hue="house_age_years",
           ci=None,
           legend_out=False)

# Add a scatter plot for prediction_data
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=prediction_data,
                hue="house_age_years",
                legend=False)

plt.show()

# Print the coeff. of determination for mdl_all_ages
print("R-squared for mdl_all_ages: ", mdl_all_ages.rsquared)

# Print the coeff. of determination for mdl_0_to_15
print("R-squared for mdl_0_to_15: ", mdl_0_to_15.rsquared)

# Print the coeff. of determination for mdl_15_to_30
print("R-squared for mdl_15_to_30: ", mdl_15_to_30.rsquared)

# Print the coeff. of determination for mdl_30_to_45
print("R-squared for mdl_30_to_45: ", mdl_30_to_45.rsquared)

# Print the RSE for mdl_all_ages
print("RSE for mdl_all_ages: ", np.sqrt(mdl_all_ages.mse_resid))

# Print the RSE for mdl_0_to_15
print("RSE for mdl_0_to_15: ", np.sqrt(mdl_0_to_15.mse_resid))

# Print the RSE for mdl_15_to_30
print("RSE for mdl_15_to_30: ", np.sqrt(mdl_15_to_30.mse_resid))

# Print the RSE for mdl_30_to_45
print("RSE for mdl_30_to_45: ", np.sqrt(mdl_30_to_45.mse_resid))

# What is an interaction?

# for example, the effect of length on the expected mass is different for different species
# i.e. length and species interact
# the effect of one explanatory variable on the expected response changes depending on the value
    # of another explanatory variable

# No interactions
# response ~ expl_var1 + expl_var2

# with interactions (implicit) (outcome same as below)
# response ~ expl_var1 * expl_var2

# with ineractions (explicit) (outcome same as above)
# response ~ expl_var1 * expl_var2 + expl_var1:expl_var2

# Running the model
mdl_mass_vs_both = ols("mass_g ~ length_cm * species", data=fish).fit()
print(mdl_mass_vs_both.params)

# to get easier to understand coeffecients:
mdl_mass_vs_both_inter = ols("mass_g ~ species + species:length_cm + 0", data=fish).fit()
print(mdl_mass_vs_both_inter.params)
# this will return an intercept and slope coefficient for each species

# Exercises

# Model price vs both with an interaction using "times" syntax
mdl_price_vs_both_inter = ols("price_twd_msq ~ n_convenience * house_age_years", data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both_inter.params)

# Model price vs. both with an interaction using "colon" syntax
mdl_price_vs_both_inter = ols("price_twd_msq ~ n_convenience + house_age_years + n_convenience:house_age_years",
                              data=taiwan_real_estate).fit()

# Print the coefficients
print(mdl_price_vs_both_inter.params)

# Making predictions with interactions
from itertools import product
length_cm = np.arange(5, 61, 5)
species = fish["species"].unique()
p = product(length_cm, species)

explanatory_data = pd.DataFrame(p, columns=["length_cm", "species"])
prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_both_inter.predict(explanatory_data))

sns.lmplot(x="length_cm", y="mass_g", data=fish, hue="species", ci=None)
sns.scatterplot(x="length_cm", y="mass_g", data=prediction_data, hue="species")
plt.show()

# Manually calculating the predictions
coeffs = mdl_mass_vs_both_inter.params 
ic_bream, ic_perch, ic_pike, ic_raoch, slope_bream, slope_perch, slope_pike, slope_roach = coeffs

conditions = [explanatory_data["species"] == "Bream",
        explanatory_data["species"] == "Perch",
        explanatory_data["species"] == "Pike",
        explanatory_data["species"] == "Roach"]

ic_choices = [ic_bream, ic_perch, ic_pike, ic_raoch]
intercept = np.select(conditions, ic_choices)

slope_choices = [slope_bream, slope_perch, slope_pike, slope_roach]
slope = np.select(conditions, slope_choices)

prediction_data = explanatory_data.assign(
    mass_g = interecpt + slope * explanatory_data["length_cm"]
)

# Exercises
# Create n_convenience as an array of numbers from 0 to 10
n_convenience = np.arange(0,11)

# Extract the unique values of house_age_years
house_age_years = taiwan_real_estate["house_age_years"].unique()

# Create p as all combinations of values of n_convenience and house_age_years
p = product(n_convenience, house_age_years)

# Transform p to a DataFrame and name the columns
explanatory_data = pd.DataFrame(p, columns=["n_convenience", "house_age_years"])

# Print it
print(explanatory_data)

# Add predictions to the DataFrame
prediction_data = explanatory_data.assign(price_twd_msq = mdl_price_vs_both_inter.predict(explanatory_data))

# Plot the trend lines of price_twd_msq vs. n_convenience colored by house_age_years
sns.lmplot(x="n_convenience",
           y="price_twd_msq",
           data=taiwan_real_estate,
           hue="house_age_years",
           ci=None)

# Add a scatter plot for prediction_data
sns.scatterplot(x="n_convenience",
                y="price_twd_msq",
                data=prediction_data, 
                hue="house_age_years",
                legend=False)

# Show the plot
plt.show()

# Get the coefficients from mdl_price_vs_both_inter
coeffs = mdl_price_vs_both_inter.params

# Assign each of the elements of coeffs
ic_0_15, ic_15_30, ic_30_45, slope_0_15, slope_15_30, slope_30_45 = coeffs

# Create conditions
conditions = [
    explanatory_data["house_age_years"] == "0 to 15",
    explanatory_data["house_age_years"] == "15 to 30",
    explanatory_data["house_age_years"] == "30 to 45"
]

# Create intercept_choices
intercept_choices = [ic_0_15, ic_15_30, ic_30_45]

# Create slope_choices
slope_choices = [slope_0_15, slope_15_30, slope_30_45]

# Create intercept and slope
intercept = np.select(conditions, intercept_choices)
slope = np.select(conditions, slope_choices)

# Create prediction_data with columns intercept and price_twd_msq
prediction_data = explanatory_data.assign(price_twd_msq = intercept + slope * explanatory_data["n_convenience"])

# Print it
print(prediction_data)

# Simpson's Paradox
# when the trend of a model on the whole dataset is very different from the trends shown by models on the
# subsets of the dataset
# helpful to visualize the data set first (to see trends among subsets)
# you can't choose the best model in general - it depends on the data set and question you want to answer
# you should be articulating a question before fitting models
# resolving the model disagreements is messy
# often the model with the groups is more insightful
# or may reveal that there are other hidden explanatory variables you need to discover

# Exercises

# Take a glimpse at the dataset
print(auctions.info())

# Model price vs. opening bid using auctions
mdl_price_vs_openbid = ols("price ~ openbid", data = auctions).fit()

# See the result
print(mdl_price_vs_openbid.params)

# Plot the scatter plot pf price vs. openbid with a linear trend line
sns.regplot(x="openbid", y="price", data=auctions, ci=None, scatter_kws={'alpha':0.5})

# Show the plot
plt.show()

# Fit linear regression of price vs. opening bid and auction type, with an interaction, without intercept
mdl_price_vs_both = ols("price ~ auction_type + openbid:auction_type + 0", data=auctions).fit()

# See the result
print(mdl_price_vs_both.params)

# Fit linear regression of price vs. opening bid and auction type, with an interaction, without intercept
mdl_price_vs_both = ols("price ~ auction_type + openbid:auction_type + 0", data=auctions).fit()

# Using auctions, plot price vs. opening bid colored by auction type as a scatter plot with linear regr'n trend lines
sns.lmplot(x="openbid", y="price", data=auctions, hue="auction_type", ci=None)

# Show the plot
plt.show()

# Two numeric explantory variables
