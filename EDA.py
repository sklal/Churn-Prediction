# EDA for Banking data to predict the propensity to churn in customers


# Importing Libraries
# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set(style="white")

# Data Loading and basic checks
# %%
df = pd.read_csv("D:\\EDA_Churn\\churn_prediction.csv")


def dfChkBasics(dframe):
    cnt = 1

    try:
        print(str(cnt)+': info(): ')
        cnt += 1
        print(dframe.info())
    except:
        pass

    print(str(cnt)+': describe(): ')
    cnt += 1
    print(dframe.describe())

    print(str(cnt)+': dtypes: ')
    cnt += 1
    print(dframe.dtypes)

    try:
        print(str(cnt)+': columns: ')
        cnt += 1
        print(dframe.columns)
    except:
        pass

    print(str(cnt)+': head() -- ')
    cnt += 1
    print(dframe.head())

    print(str(cnt)+': shape: ')
    cnt += 1
    print(dframe.shape)


dfChkBasics(df)

# Observations
# Numerical Features
# Customer ID here is just an id variable identifying a unique customer and has values between 1 and 30301
# On average, a customer from this set has been with the bank for 2400 days or around 6.5 years
# On average, a customer has less than 1 dependent and has an average age of 48 years
# A general trend on variables which are related to balances have a wide range with huge outliers, it will key to observe these outliers
# Most of the customers lie in category 2 or 3 for net worth and have on an average done the last transaction 70 days ago. Now the high net worth customers (Category) must have high credit, debit and balance values.


# Targert exploration
# %%
ax = sns.catplot(y="churn", kind="count", data=df,
                 height=2.6, aspect=2.5, orient='h')
df['churn'].value_counts(normalize=True)

# Customer Net worth Category & Balance Features
# %%
cols = ['current_balance',
        'previous_month_end_balance', 'average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2',
        'current_month_credit', 'previous_month_credit', 'current_month_debit', 'previous_month_debit',
        'current_month_balance', 'previous_month_balance']

df.groupby(['customer_nw_category'])[cols].mean()

# There is clear consistency here as mean values of balance features and the credit/debit features have higher values for net worth category 1 and lower value for net worth categories 2 & 3

# Balance & Credit/Debit Features
# %%
sns.distplot(df['current_month_balance'], kde=False)
plt.show()

# Due to the huge outliers in both positive and negative directions, it is very difficult to derive insights from this plot.
# In this case, we could convert such columns to log and then check the distributions.
# However, since there are negative values, it cannot be a direct log conversion as log of negative numbers is not defined.
# To tackle this, we add a positive constant within the log as a correction

# To account for negative values we add a constant value within log
temp = np.log(df['current_month_balance'] + 5000)

sns.distplot(temp, kde=False, bins=100)
plt.show()

# we can see more clearly that this is a right skewed feature and we have much more clarity on its distribution

# %%
# Numerical Features
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
xmin = 7
xmax = 16
# Current Month Average Balance
# To account for negative values we add a constant value within log
temp = np.log(df['current_month_balance'] + 5000)
ax1.set_xlim([xmin, xmax])
ax1.set(xlabel='log of average balance of current month')
sns.distplot(temp, kde=False, bins=200, ax=ax1)


# Previous month average balance
# To account for negative values we add a constant value within log
temp = np.log(df['previous_month_balance'] + 5000)
ax2.set_xlim([xmin, xmax])
ax2.set(xlabel='log of average balance of previous month')
sns.distplot(temp, kde=False, bins=200, ax=ax2)

plt.show()

# As expected the average monthly balance for both months are quite similar and have right skewed histograms as shown

# Current Balance today vs Average Monthly Balance in current month
# %%
# Numerical Features
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
xmin = 7
xmax = 16
# Current Month Average Balance
# To account for negative values we add a constant value within log
temp = np.log(df['current_month_balance'] + 6000)
ax1.set_xlim([xmin, xmax])
#ax1.set(xlabel='log of average balance of current month')
sns.distplot(temp, kde=False, bins=200, ax=ax1)


# Current End of month average balance
# To account for negative values we add a constant value within log
temp = np.log(df['current_balance'] + 6000)
ax2.set_xlim([xmin, xmax])
#ax2.set(xlabel='log of month end balance of current  month')
sns.distplot(temp, kde=False, bins=200, ax=ax2)


plt.show()

# Here, we can see that the distribution for both lie in almost the same interval, however, there are larger number of values for current balance just below 9 which might have been contributed by the churning customers.
# It might be a good idea to create a feature which is the difference of these 2 variables during the model building process.

# Churn vs Current & Previous month balances
# %%
balance_cols = ['current_balance', 'previous_month_end_balance',
                'current_month_balance', 'previous_month_balance']
df1 = pd.DataFrame()

for i in balance_cols:
    df1[str('log_') + i] = np.log(df[i] + 5000)

log_balance_cols = df1.columns

df1.head()

df1['churn'] = df['churn']

#sns.pairplot(df1,vars=log_balance_cols, hue = 'churn',plot_kws={'alpha':0.1})
df1_no_churn = df1[df1['churn'] == 0]
sns.pairplot(df1_no_churn, vars=log_balance_cols, plot_kws={'alpha': 0.1})
plt.show()

#sns.pairplot(df1,vars=log_balance_cols, hue = 'churn',plot_kws={'alpha':0.1})
df1_churn = df1[df1['churn'] == 1]
sns.pairplot(df1, vars=log_balance_cols, plot_kws={'alpha': 0.1})
plt.show()

sns.pairplot(df1, vars=log_balance_cols, hue='churn', plot_kws={'alpha': 0.1})
plt.show()

# The distribution for these features look similar. We can make the following conclusions from this:

# There is high correlation between the previous and current month balances which is expected
# The lower balances tend to have higher number of churns which is clear from the scatter plots
# Distribution for the balances are all right skewed


# %%
# Total credit and debit amounts for the current and previous can be clubbed into the same category.

cr_dr_cols = ['current_month_credit', 'previous_month_credit',
              'current_month_debit', 'previous_month_debit']
df11 = pd.DataFrame()

for i in cr_dr_cols:
    df1[str('log_') + i] = np.log(df[i])

log_dr_cr_cols = df1.columns

df1['churn'] = df['churn']

sns.pairplot(df1, vars=log_dr_cr_cols, hue='churn', plot_kws={'alpha': 0.5})
plt.show()

# Both credit and debit patterns show significant difference in distributions for churned and non churned customers.

# Bimodal distribution/Double Bell Curve shows that there are 2 different types of customers with 2 brackets of credit and debit.
#  Now, during the modeling phase, these could be considered as a seperate set of customers
# For debit values, we see that there is a significant difference in the distribution for churn and non churn and it might be turn out to be an important feature


# Average monthly balance of previous and previous to previous quarters

q_cols = ['average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2']
df1 = pd.DataFrame()

for i in q_cols:
    df1[str('log_') + i] = np.log(df[i] + 15000)

log_q_cols = df1.columns
df1['churn'] = df['churn']

# The distributions do not have much difference when it comes to churn.
# However, there are some high negative values in the previous to previous quarters due to which there appears to be a lateral shift. However, if you look at the x-axis, it is still at the same scale for both features.
# Removing the extreme outliers from the data using the 1 and 99th percentile would help us look at the correct distributions

# Remove 1st and 99th percentile and plot

df2 = df[['average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2']]

low = .01
high = .99
quant_churn = df2.quantile([low, high])
print(quant_churn)

q_cols = ['average_monthly_balance_prevQ', 'average_monthly_balance_prevQ2']
df1 = pd.DataFrame()

for i in q_cols:
    df1[str('log_') + i] = np.log(df3[i] + 17000)

log_q_cols = df1.columns
df1['churn'] = df['churn']

sns.pairplot(df1, vars=log_q_cols, hue='churn', plot_kws={'alpha': 0.5})
plt.show()

# we can clearly see that the distributions are very similar for both the variables and and non churning customers have higher average monthly balances in previous 2 quarters


# Percentage Change in Credits
# %%
change_cols = ['percent_change_credits']
df1 = pd.DataFrame()

for i in change_cols:
    df1[str('log_') + i] = np.log(df[i] + 100)

log_change_cols = df1.columns
df1['churn'] = df['churn']

sns.pairplot(df1, vars=log_change_cols, hue='churn', plot_kws={'alpha': 0.2})
plt.show()

# Percent change in credits has a very nice almost normal distribution after log transfromation and does not have significantly different distribution for both churning and non churning customers.

# Days since last transaction
# %%


def kdeplot(feature):
    plt.figure(figsize=(9, 4))
    plt.title("KDE Plot for {}".format(feature))
    ax0 = sns.kdeplot(df[df['churn'] == 0][feature].dropna(),
                      color='dodgerblue', label='Churn - 0')
    ax1 = sns.kdeplot(df[df['churn'] == 1][feature].dropna(),
                      color='orange', label='Churn - 1')


kdeplot('days_since_last_transaction')

# There is no significant difference between the distributions for churning and non churning customers when it comes to days since last transaction.


# Age
# %%
kdeplot('age')

# Similarly, age also does not significantly affect the churning rate. However, customers above 80 years of age less likely to churn

# Vintage
# %%
kdeplot('vintage')

# For most frequent vintage values, the churning customers are slightly higher, while for higher values of vintage, we have mostly non churning customers which is in sync with the age variable

# Categorical features
cat_cols = ['gender', 'occupation', 'city', 'branch_code']

for i in range(0, len(cat_cols)):
    print(str(cat_cols[i]) + " - Number of Unique Values: " +
          str(df[cat_cols[i]].nunique()))


# There are a large number of unique values for branch code and city. Gender has 2 unique values while occupation has 7

# Univariate Analysis
# %%
color = sns.color_palette()

int_level = df['gender'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Gender', fontsize=12)
plt.show()

color = sns.color_palette()

int_level = df['occupation'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Occupation', fontsize=12)
plt.show()

color = sns.color_palette()

int_level = df['city'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('City', fontsize=12)
plt.show()


df['city'].value_counts().head(20)

# Convert city variable wrt degree of number of customers
df['city_bin'] = df['city'].copy()
counts = df.city.value_counts()
df.city_bin[df['city'].isin(counts[counts > 900].index)] = 3
df.city_bin[df['city'].isin(counts[counts < 900].index) & df['city_bin'].isin(
    counts[counts >= 350].index)] = 2
df.city_bin[df['city'].isin(counts[counts < 350].index) & df['city_bin'].isin(
    counts[counts >= 100].index)] = 1
df.city_bin[df['city'].isin(counts[counts < 100].index)] = 0

df['city_bin'] = pd.to_numeric(df['city_bin'], errors='coerce')

int_level = df['city_bin'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('city bins', fontsize=12)
plt.show()

# There are 2 major categories here. Cities with more than 900 occurances and with less than 100 occurances. Similarly we can create bins for branch id and have a look.

df['branch_code'].value_counts()

# Convert city variable wrt degree of number of customers
df['branch_bin'] = df['branch_code'].copy()
counts = df.branch_code.value_counts()
df.branch_bin[df['branch_code'].isin(counts[counts >= 100].index)] = 2
df.branch_bin[df['branch_code'].isin(
    counts[counts < 100].index) & df['branch_bin'].isin(counts[counts >= 50].index)] = 1
df.branch_bin[df['branch_code'].isin(counts[counts < 50].index)] = 0

df['branch_bin'] = pd.to_numeric(df['branch_bin'], errors='coerce')

df['branch_bin'].value_counts()

int_level = df['branch_bin'].value_counts()

plt.figure(figsize=(8, 4))
sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('branch bins', fontsize=12)
plt.show()

# Bivariate Analysis
# %%


def barplot_percentages(feature):
    #fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))
    ax1 = df.groupby(feature)['churn'].value_counts(normalize=True).unstack()
    ax1.plot(kind='bar', stacked='True')
    int_level = df[feature].value_counts()

    plt.figure(figsize=(8, 4))
    sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(str(feature), fontsize=12)
    plt.show()


# Gender
barplot_percentages("gender")

# Occupation
barplot_percentages("occupation")

# Branchbins
barplot_percentages("branch_bin")

# City bins
barplot_percentages("city_bin")

# Here, we see significant difference for different occupations and certainly would be interesting to use as a feature for prediction of churn. However, city and branch codes have little difference amongst the different types of branches

# Dependents
df['dependents'][df['dependents'] > 3] = 3
barplot_percentages("dependents")

# Most customers have no dependents and hence this variable in itself has low variance so it is of little significance

# Customer Net worth Category
barplot_percentages("customer_nw_category")

# Not much difference in customer net worth category when it comes to churn


# Correlation Heatmap
# %%
plt.figure(figsize=(12, 6))
df.drop(['customer_id'],
        axis=1, inplace=True)
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                 linewidths=.2, cmap="YlGnBu")

# The balance features are highly correlated as can be seen from the plot
# Other variables have correlations but on the lower side
# Debit values have the highest correlation amongst the balance features
# Interestingly vintage has a considerable correlation with all the balance features which actually makes sense since older customers will tend to have higher balance


# Conclusions
# Average customer Profile
# Overall a customer at this bank:

# has no dependents
# has been a customer for last 6 years
# predominantly male
# either self employed or salaried customer


# Conclusion for Churn
# From the sample, around 17% customers are churning
# Current balance and average monthly balance values have a left skewed distribution as observed from the histogram
# No significant difference in distributions for average monthly balance and month end balances
# Bimodal distribution/Double Bell Curve shows that there are 2 different types of customers with 2 brackets of credit and debit. Now, during the modeling phase, these could be considered as a seperate set of customers
# For debit values, we see that there is a significant difference in the distribution for churn and non churn and it might be turn out to be an important feature
# For most frequent vintage values, the churning customers are slightly higher, while for higher values of vintage, we have mostly non churning customers which is in sync with the age variable
# Gender does not look like a very significant variable as the ratio of churned customers and others is very similar
# Self Employed and salaried have higher churn rate and are the most frequently occuring categories.
# Not much difference in customer net worth category when it comes to churn
