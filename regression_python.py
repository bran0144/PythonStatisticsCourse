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

# plotting mass vs. length
sns.regplot(x="length_cm", y="mass_g", data=bream, ci=None)
plt.show()

mdl_mass_vs_length = ols("mass_g ~ length_cm", data=bream).fit()
print(mdl_mass_vs_length.params)

explanatory_data = pd.DataFrame({"legnth_cm": np.arange(20,41)})
print(mdl_mass_vs_length.predict(explanatory_data))
# this returns a df with both explanatory variable and the predicted response
prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_length.predict(explanatory_data))

fig = plt.figure()
sns.regplot(x="length_cm", y="mass_g", ci=None, data=bream)
sns.scatterplot(x="length_cm", y="mass_g", data=prediction_data, color="red", marker="s")
plt.show()

# Extrapolating
# making predictions outside the range of observed data

little_bream = pd.DataFrame({"legnth_cm": [10]})
pred_little_bream = little_bream.assign(mass_g=mdl_mass_vs_length.predict(little_bream))
print(pred_little_bream)

# this doesn't work. It return a negative value. Extrapoliation sometimes will give you misleading results

# Exercises

# Import numpy with alias np
import numpy as np

# Create the explanatory_data 
explanatory_data = pd.DataFrame({'n_convenience': np.arange(0,11)})

# Print it
print(explanatory_data)

# Use mdl_price_vs_conv to predict with explanatory_data, call it price_twd_msq
price_twd_msq = mdl_price_vs_conv.predict(explanatory_data)

# Create prediction_data
prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_conv.predict(explanatory_data))

# Print the result
print(prediction_data)

# Create a new figure, fig
fig = plt.figure()

sns.regplot(x="n_convenience",
            y="price_twd_msq",
            data=taiwan_real_estate,
            ci=None)
# Add a scatter plot layer to the regplot
sns.scatterplot(x='n_convenience', y='price_twd_msq', data=prediction_data, color="red")

# Show the layered plot
plt.show()

# Model Objects

# fitted values (takes explanatory variable columns from data, then you can feed them to a predict function)
print(mdl_mass_vs_length.fittedvalues)

# residuals - actual response values minue predicted response values (measure of inaccuracy)
print(mdl_mass_vs_length.resid)
# one residual for each row of the dataset

# details and extended printout
mdl_mass_vs_length.summary()
# what's in the printout?
# dependent variables, type of regression, method, # of obs, performance metrics
# details of coefficients (intercepts, std, p values, etc)
# diagnostic statistics 

# Exercises

# Get the coefficients of mdl_price_vs_conv
coeffs = mdl_price_vs_conv.params

# Get the intercept
intercept = coeffs[0]

# Get the slope
slope = coeffs[1]

# Manually calculate the predictions
price_twd_msq = slope * explanatory_data + intercept
print(price_twd_msq)

# Compare to the results from .predict()
print(price_twd_msq.assign(predictions_auto=mdl_price_vs_conv.predict(explanatory_data)))

# Regression to the mean

# response value = fitted value + residual
# residuals - can be due to problems with your model and/or fundamental randomness
# extreme responses are often due to randomness
# regression to the man - extreme cases don't persist over time (they will head toward the mean)

fig = plt.figure()
sns.scatterplot(x="father_height_cm", y="son_height_cm", data=father_son)
sns.regplot(x="father_height_cm", y="son_height_cm", data=father_son, ci=None, line_kqs={"color":"black"})
plt.axline(xy1=(150,150), slope=1, linewidth=2, color="green")
plt.axis("equal")
plt.show()

mdl_son_vs_father = ols("son_height_cm ~ father_height_cm", data=father_son).fit()
print(mdl_son_vs_father.params)

really_tall_father = pd.DataFrame({"father_heigh_cm": [190]})
mdl_son_vs_father.predict(really_tall_father)


# Exercises

# Create a new figure, fig
fig = plt.figure()

# Plot the first layer: y = x
plt.axline(xy1=(0,0), slope=1, linewidth=2, color="green")

# Add scatter plot with linear regression trend line
sns.regplot(x="return_2018", y="return_2019", data=sp500_yearly_returns, ci=None)

# Set the axes so that the distances along the x and y axes look the same
plt.axis("equal")

# Show the plot
plt.show()

# Run a linear regression on return_2019 vs. return_2018 using sp500_yearly_returns
mdl_returns = ols("return_2019 ~ return_2018", data=sp500_yearly_returns).fit()

# Print the parameters
print(mdl_returns)

mdl_returns = ols("return_2019 ~ return_2018", data=sp500_yearly_returns).fit()

# Create a DataFrame with return_2018 at -1, 0, and 1 
explanatory_data = pd.DataFrame({"return_2018": [-1,0,1]})

# Use mdl_returns to predict with explanatory_data
print(mdl_returns.predict(explanatory_data))

# Transforming Variables
perch = fish[fish['species'] == "Perch"]
sns.regplot(x="length_cm", y="mass_g", data=perch, ci=None)
plt.show()

# not a linear relationship, so you may need to transform the data
perch['length_cm_cubed'] = perch['length_cm'] ** 3
sns.regplot(x="length_cm_cubed", y="mass_g", data=perch, ci=None)
plt.show()

mdl_perch = old('mass_g ~ length_cm_cubed', data=perch).fit()
mdl_perch.params

explanatory_data = pd.DataFrame({'length_cm_cubed': np.arange(10,41,5)**3,
        'length_cm': np.arange(10,41,5)})

prediction_data = explanatory_data.assign(mass_g=mdl_perch.predict(explanatory_data))
print(prediction_data)

fig = plt.figure()
sns.regplot(x="length_cm_cubed", y="mass_g", data=perch, ci=None)
sns.scatterplot(data=prediction_data, x="length_cm_cubed", y="mass_g", color="red", marker="s")

fig = plt.figure()
sns.regplot(x="length_cm", y="mass_g", data=perch, ci=None)
sns.scatterplot(data=prediction_data, x="length_cm", y="mass_g", color="red", marker="s")

# facebook impressions vs spent
sns.regplot(x="spend_usd", y="n_impressions", data=ad_conversion, ci=None)

# transform as sqrt vs sqrt
ad_conversion['sqrt_spent_usd'] = np.sqrt(ad_conversion['spent_usd'])
ad_conversion['sqrt_n_impressions'] = np.sqrt(ad_conversion['n_impressions'])
sns.regplot(x="sqrt_spent_usd", y="sqrt_n_impressions", data=ad_conversion, ci=None)

# common transformation when your data has a right skewed distribution
mdl_ad = ols("sqrt_n_impressions ~ sqrt_spent_usd", data=ad_conversion).fit()
explanatory_data = pd.DataFrame({"sqrt_spent_usd": np.sqrt(np.arange(0,601,100)),
        "spent_usd": np.arange(0,601,100)})

# back transformation = undoing the trasnformation in the response
prediction_data = explanatory_data.assign(sqrt_n_impressions=mdl_ad.predict(explanatory_data),
        n_impressions=mdl_ad.predict(explanatory_data)**2)

print(prediction_data)

# Exercises

# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

plt.figure()

# Plot using the transformed variable
sns.regplot(x="sqrt_dist_to_mrt_m", y="price_twd_msq", data=taiwan_real_estate, ci=None)
plt.show()

# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

# Run a linear regression of price_twd_msq vs. square root of dist_to_mrt_m using taiwan_real_estate
mdl_price_vs_dist = ols("price_twd_msq ~ sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

# Print the parameters
mdl_price_vs_dist.params

# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

# Run a linear regression of price_twd_msq vs. sqrt_dist_to_mrt_m
mdl_price_vs_dist = ols("price_twd_msq ~ sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

explanatory_data = pd.DataFrame({"sqrt_dist_to_mrt_m": np.sqrt(np.arange(0, 81, 10) ** 2),
                                "dist_to_mrt_m": np.arange(0, 81, 10) ** 2})

# Create prediction_data by adding a column of predictions to explantory_data
prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_dist.predict(explanatory_data)
)

# Print the result
print(prediction_data)

# Create sqrt_dist_to_mrt_m
taiwan_real_estate["sqrt_dist_to_mrt_m"] = np.sqrt(taiwan_real_estate["dist_to_mrt_m"])

# Run a linear regression of price_twd_msq vs. sqrt_dist_to_mrt_m
mdl_price_vs_dist = ols("price_twd_msq ~ sqrt_dist_to_mrt_m", data=taiwan_real_estate).fit()

# Use this explanatory data
explanatory_data = pd.DataFrame({"sqrt_dist_to_mrt_m": np.sqrt(np.arange(0, 81, 10) ** 2),
                                "dist_to_mrt_m": np.arange(0, 81, 10) ** 2})

# Use mdl_price_vs_dist to predict explanatory_data
prediction_data = explanatory_data.assign(
    price_twd_msq = mdl_price_vs_dist.predict(explanatory_data)
)

fig = plt.figure()
sns.regplot(x="sqrt_dist_to_mrt_m", y="price_twd_msq", data=taiwan_real_estate, ci=None)

# Add a layer of your prediction points
sns.scatterplot(data=prediction_data, x="sqrt_dist_to_mrt_m", y="price_twd_msq", color="red")
plt.show()

# Create qdrt_n_impressions and qdrt_n_clicks
ad_conversion["qdrt_n_impressions"] = ad_conversion["n_impressions"] ** .25
ad_conversion["qdrt_n_clicks"] = ad_conversion["n_clicks"] ** .25

plt.figure()

# Plot using the transformed variables
sns.regplot(x="qdrt_n_impressions", y="qdrt_n_clicks", data=ad_conversion, ci=None)
plt.show()

ad_conversion["qdrt_n_impressions"] = ad_conversion["n_impressions"] ** 0.25
ad_conversion["qdrt_n_clicks"] = ad_conversion["n_clicks"] ** 0.25

# Run a linear regression of your transformed variables
mdl_click_vs_impression = ols("qdrt_n_clicks ~ qdrt_n_impressions", data=ad_conversion).fit()

ad_conversion["qdrt_n_impressions"] = ad_conversion["n_impressions"] ** 0.25
ad_conversion["qdrt_n_clicks"] = ad_conversion["n_clicks"] ** 0.25

mdl_click_vs_impression = ols("qdrt_n_clicks ~ qdrt_n_impressions", data=ad_conversion, ci=None).fit()

explanatory_data = pd.DataFrame({"qdrt_n_impressions": np.arange(0, 3e6+1, 5e5) ** .25,
                                 "n_impressions": np.arange(0, 3e6+1, 5e5)})

# Complete prediction_data
prediction_data = explanatory_data.assign(
    qdrt_n_clicks = mdl_click_vs_impression.predict(explanatory_data)
)

# Print the result
print(prediction_data)

# Back transform qdrt_n_clicks
prediction_data["n_clicks"] = prediction_data["qdrt_n_clicks"] ** 4
print(prediction_data)

# Back transform qdrt_n_clicks
prediction_data["n_clicks"] = prediction_data["qdrt_n_clicks"] ** 4

# Plot the transformed variables
fig = plt.figure()
sns.regplot(x="qdrt_n_impressions", y="qdrt_n_clicks", data=ad_conversion, ci=None)

# Add a layer of your prediction points
sns.scatterplot(data=prediction_data, x="qdrt_n_impressions", y="qdrt_n_clicks", color="red")
plt.show()

# Quantifying model fit
mdl_bream = ols("mass_g ~ length_cm", data=bream).fit()
print(mdl_bream.summary())

print(mdl_bream.rsquared)

coeff_determination = bream['length_cm'].corr(bream['mass_g']) ** 2
print(coeff_determination)

# RSE - residual standard error
"measure of the typical size of the residuals"
# same unit as the response varaible
# MSE- Mean squared error - squared residual error
# summary method does not continer RSE
mse = mdl_bream.mse_resid 
rse = np.sqrt(mse)

# to calculatse rse
resideals_sq = mdl_bream/resid **2
resid_sum_of_sq = sum(resideals_sq)
deg_freedom = len(bream.index) -2
# degrees of freedom = # of obs  minus # of model coefficients
rse = np.sqrt(resid_sum_of_sq/deg_freedom)

# rmse = root mean squared error
n_obs = len(bream.index)
rmse = mp.sqrt(resid_sum_of_sq/n_obs)

# Exercises
# Print a summary of mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.summary())

# Print a summary of mdl_click_vs_impression_trans
print(mdl_click_vs_impression_trans.summary())

# Print the coeff of determination for mdl_click_vs_impression_orig
print(mdl_click_vs_impression_orig.rsquared)

# Print the coeff of determination for mdl_click_vs_impression_trans
print(mdl_click_vs_impression_trans.rsquared)

# Calculate mse_orig for mdl_click_vs_impression_orig
mse_orig = mdl_click_vs_impression_orig.mse_resid

# Calculate rse_orig for mdl_click_vs_impression_orig and print it
rse_orig = np.sqrt(mse_orig)
print("RSE of original model: ", rse_orig)

# Calculate mse_trans for mdl_click_vs_impression_trans
mse_trans = mdl_click_vs_impression_trans.mse_resid

# Calculate rse_trans for mdl_click_vs_impression_trans and print it
rse_trans = np.sqrt(mse_trans)
print("RSE of transformed model: ", rse_trans)

# Visualizing model fit

# plotting the residuals helps to see model fit
# QQ-Plot - if points track along line, they are normally distributed
# scale location plot - shows the sqrt of the standardized residuals vs the fitted values
# shows whether the size of the residuals get bigger or smaller

sns.residplot(x="length_cm", y="mass_g", data=bream, lowess=True)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")

from statsmodels.api import qqplot 
qqplot(data=mdl_bream.resid, fit=True, line="45")

# scale location plot

model_norm_residuals_bream = mdl_bream.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt_bream = np.sqrt(np.abs(model_norm_residuals_bream))
sns.regplot(x=mdl_bream.fittedvalues, y=model_norm_residuals_abs_sqrt_bream, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")

# Exercises

# Plot the residuals vs. fitted values
sns.residplot(x="n_convenience", y="price_twd_msq", data=taiwan_real_estate, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")

# Show the plot
plt.show()

# Preprocessing steps
model_norm_residuals = mdl_price_vs_conv.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

# Create the scale-location plot
sns.regplot(x=mdl_price_vs_conv.fittedvalues, y=model_norm_residuals_abs_sqrt, ci=None, lowess=True)
plt.xlabel("Fitted values")
plt.ylabel("Sqrt of abs val of stdized residuals")

# Show the plot
plt.show()

# Outliers

roach = fish[fish['species'] == 'Roach']

sns.regplot(x="length_cm", y="mass_g", data=roach, ci=None)
plt.show()

roach["extreme_l"] = ((roach['legnth_cm'] < 15) | (raoch['length_cm']> 26))
fig = plt.figure()
sns.regplot(x="length_cm", y="mass_g", data=roach, ci=None)
sns.scatterplot(x="length_cm", y="mass_g", hue="extreme_l", data=roach)

roach["extreme_m"] = roach['mass_g'] < 1
fig = plt.figure()
sns.regplot(x="length_cm", y="mass_g", data=roach, ci=None)
sns.scatterplot(x="length_cm", y="mass_g", hue="extreme_l", style="extreme_m", data=roach)

# Leverage - how extreme your explanatory variables are
# Influence - how much the model would change if you left the observation out of the dataset when modeling

mdl_roach = ols("mass_g ~ length_cm", data=roach).fit()
summary_raoch = mdl_roach.get_influence().summary_frame()

roach['leverage'] = summary_roach['hat_diag']

# Cook's distance - the most common measure of influence
roach['cooks_dist'] = summary_raoch["cooks_d"]

# most influenial
print(roach.sort_values('cooks_dist', ascending=False))

roach_not_short = roach[roach['length_cm'] != 12.9]

sns.regplot(x="length_cm", y="mass_g", data=roach, ci=None, line_kws={"color":"green"})
sns.regplot(x="length_cm", y="mass_g", data=roach_not_short, ci=None, line_kws={"color":"red"})

# Exercises
# Create summary_info
summary_info = mdl_price_vs_dist.get_influence().summary_frame()

# Add the hat_diag column to taiwan_real_estate, name it leverage
taiwan_real_estate["leverage"] = summary_info['hat_diag']

# Sort taiwan_real_estate by leverage in descending order and print the head
print(taiwan_real_estate.sort_values('leverage', ascending=False).head())

# Add the cooks_d column to taiwan_real_estate, name it cooks_dist
taiwan_real_estate['cooks_dist'] = summary_info['cooks_d']

# Sort taiwan_real_estate by cooks_dist in descending order and print the head.
print(taiwan_real_estate.sort_values('cooks_dist', ascending=False).head())

# Logistic Regression
mdl_churn_vs_recency_lm = ols("has_churned ~ time_since_last_purchase", data=churn).fit()
# linear returns impossible values and isn't helpful when your data isn't linear
# logistic models follow a logitics (S shaped curve rather than a line)

from statsmodels.formula.api import logit 
mdl_churn_vs_recency_logit = logit("has_churned ~ time_since_last_purchase", data=churn).fit()
print(mdl_churn_vs_recency_logit.params)

sns.regplot(x="time_since_last_purchased", y="has_churned", data=churn, ci=None, logistic=True)
plt.axline(xy1=(0, intercept), slope=slope, color="black")
plt.show()


# Exercises

# Create the histograms of time_since_last_purchase split by has_churned
sns.displot(data=churn, x="time_since_last_purchase", col="has_churned")

plt.show()

# Redraw the plot with time_since_first_purchase
sns.displot(data=churn, x="time_since_first_purchase", col="has_churned")

plt.show()

# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase", y="has_churned", data=churn, ci=None, logistic=True,
            line_kws={"color": "red"})

plt.show()

# Draw a linear regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn, 
            ci=None,
            line_kws={"color": "red"})

# Draw a logistic regression trend line and a scatter plot of time_since_first_purchase vs. has_churned
sns.regplot(x="time_since_first_purchase", y="has_churned", data=churn, ci=None, logistic=True,
            line_kws={"color": "blue"})

plt.show()

# Import logit
from statsmodels.formula.api import logit

# Fit a logistic regression of churn vs. length of relationship using the churn dataset
mdl_churn_vs_relationship = logit("has_churned ~ time_since_first_purchase", data=churn).fit()

# Print the parameters of the fitted model
print(mdl_churn_vs_relationship.params)

# makind predictions

mdl_recency = logit("has_churned ~ time_since_last_purchase", data=churn).fit()

explanatory_data = pd.DataFrame({"time_since_last_purchase": np.arange(-1,6.25, 0.25)})
prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))

sns.regplot(x="time_since_last_purchase", y="has_churned", data=churn, ci=None, logistic=True)
sns.scatterplot(x="time_since_last_purchase", y="has_churned", data=prediction_data, color="red")
plt.show()

prediction_data = explanatory_data.assign(has_churned = mdl_recency.predict(explanatory_data))

prediction_data['most_likely_outcome'] = np.round(prediction_data['has_churned'])

sns.regplot(x="time_since_last_purchase", y="has_churned", data=churn, ci=None, logistic=True)
sns.scatterplot(x="time_since_last_purchase", y="has_churned", data=prediction_data, color="red")
plt.show()

# Odds ratios
# odds_ratio = probability that something happens / 1 - probability

prediction_data['odds_ratio'] = prediction_data['has_churned'] / (1 - prediction_data['has_churned'])

sns.lineplot(x="time_since_last_purchase", y="odd_ratio", data=prediction_data)
plt.axhline(y=1, linetyle="dotted")
plt.yscale("log")
plt.show()

prediction_data['log_odds_ratio'] = np.log(prediction_data['odd_ratio'])


# Exercises:

# Create prediction_data
prediction_data = explanatory_data.assign(has_churned=mdl_churn_vs_relationship.predict(explanatory_data)
  
)

# Print the head
print(prediction_data.head())

# Create prediction_data
prediction_data = explanatory_data.assign(
    has_churned = mdl_churn_vs_relationship.predict(explanatory_data)
)

fig = plt.figure()

# Create a scatter plot with logistic trend line
sns.regplot(x="time_since_first_purchase", y="has_churned", data=churn, ci=None, logistic=True)
# Overlay with prediction_data, colored red

sns.scatterplot(x="time_since_first_purchase", y="has_churned", data=prediction_data, color="red")

plt.show()

# Update prediction data by adding most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(prediction_data['has_churned'])

# Print the head
print(prediction_data.head())

# Update prediction data by adding most_likely_outcome
prediction_data["most_likely_outcome"] = np.round(prediction_data["has_churned"])

fig = plt.figure()

# Create a scatter plot with logistic trend line (from previous exercise)
sns.regplot(x="time_since_first_purchase",
            y="has_churned",
            data=churn,
            ci=None,
            logistic=True)

# Overlay with prediction_data, colored red
sns.scatterplot(x="time_since_first_purchase",
                y="most_likely_outcome",
                data=prediction_data,
                color="red")

plt.show()

# Update prediction data with odds_ratio
prediction_data["odds_ratio"] = prediction_data['has_churned'] / (1 - prediction_data['has_churned'])

# Print the head
print(prediction_data.head())

# Update prediction data with odds_ratio
prediction_data["odds_ratio"] = prediction_data["has_churned"] / (1 - prediction_data["has_churned"])

fig = plt.figure()

# Create a line plot of odds_ratio vs time_since_first_purchase
sns.lineplot(x="time_since_first_purchase", y="odds_ratio", data=prediction_data)

# Add a dotted horizontal line at odds_ratio = 1
plt.axhline(y=1, linestyle="dotted")

plt.show()

# Update prediction data with log_odds_ratio
prediction_data['log_odds_ratio'] = np.log(prediction_data['odds_ratio'])

# Print the head
print(prediction_data.head())

# Update prediction data with log_odds_ratio
prediction_data["log_odds_ratio"] = np.log(prediction_data["odds_ratio"])

fig = plt.figure()

# Update the line plot: log_odds_ratio vs. time_since_first_purchase
sns.lineplot(x="time_since_first_purchase",
             y="log_odds_ratio",
             data=prediction_data)

# Add a dotted horizontal line at log_odds_ratio = 0
plt.axhline(y=0, linestyle="dotted")

plt.show()

# Quantifying logistic regression fit
# confusion matrix - counts of each possible outcome (correct, false negative, false positive)

actual_response = churn["has_churned"]
predicted_response = np.round(mdl_recency.predict())
outcomes = pd.DataFrame({'actual_response': actual_response, 'predicted_response': predicted_response})

print(outcomes.value_counts(sort=False))

conf_matrix= mdl_recency.pred_table()
print(conf_matrix)

from statsmodels.graphics.mosaicplot import mosaic 

# accuracy - proportion of correct predictions
# # of true negatives plus the true positives divided by total number of observations

# sensitivity - proportion of observattions where the actual response was true where the
# model also predicted that they were true
# # of true positives divided by the false negatives and true positives

# specificity - proportion of obs where actual repsonse was false where model predicted false
# # of true negatives divided by teh sum of true negatives and false positives

# Exercises

# Get the actual responses
actual_response = churn['has_churned']

# Get the predicted responses
predicted_response = np.round(mdl_churn_vs_relationship.predict())

# Create outcomes as a DataFrame of both Series
outcomes = pd.DataFrame({'actual_response': actual_response,
                         'predicted_response': predicted_response})

# Print the outcomes
print(outcomes.value_counts(sort = False))

# Import mosaic from statsmodels.graphics.mosaicplot
from statsmodels.graphics.mosaicplot import mosaic 

# Calculate the confusion matrix conf_matrix
conf_matrix = mdl_churn_vs_relationship.pred_table()

# Print it
print(conf_matrix)

# Draw a mosaic plot of conf_matrix
mosaic(conf_matrix)
plt.show()

# Extract TN, TP, FN and FP from conf_matrix
TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]

# Calculate and print the accuracy
accuracy = (TN + TP) / (TN + FN + FP + TP)
print("accuracy: ", accuracy)

# Calculate and print the sensitivity
sensitivity = TP / (TP + FN)
print("sensitivity: ", sensitivity)

# Calculate and print the specificity
specificity = TN / (TN + FP)
print("specificity: ", specificity)

