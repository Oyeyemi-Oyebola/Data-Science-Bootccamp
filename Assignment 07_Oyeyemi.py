#!/usr/bin/env python
# coding: utf-8

# ## AccelerateAI - Python for Data Science - Assignment 07
# ### Multiple Linear Regression

# In[103]:


## creating function to get model statistics
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Q1. MLR Stepwise Regression – Household Expense
# 500 household were surveyed on their monthly expenses. The data is in the file
# MLR_MonthlyExpense.

# In[93]:


MLR_MonthlyExpense = pd.read_csv("C:\\Users\\theco\\MLR_MonthlyExpense.csv")


# In[94]:


MLR_MonthlyExpense.head()


# In[75]:


# Lets look at relationship of Monthly Payment with the numerical predictors
sns.pairplot(MLR_MonthlyExpense,     
             x_vars=["Family Size", "Sector No", "Rent", "Own", "Income", "Utilities", "Debt"],
             y_vars=["Monthly Payment"])


# In[95]:


X = MLR_MonthlyExpense.drop(["Household", "Rent"], axis=1)


# In[96]:


X.head()


# In[ ]:


# Make salary  numeric datatype
MLR_MonthlyExpense["Monthly Payment"] = MLR_MonthlyExpense["Monthly Payment"].apply(lambda x: int(x.replace('$','').replace(',',''))) 


# In[98]:


# Make salary  numeric datatype
MLR_MonthlyExpense["Income"] = MLR_MonthlyExpense["Income"].apply(lambda x: int(x.replace('$','').replace(',',''))) 


# In[99]:


# Make salary  numeric datatype
MLR_MonthlyExpense["Utilities"] = MLR_MonthlyExpense["Utilities"].apply(lambda x: int(x.replace('$','').replace(',',''))) 


# In[100]:


# Make salary  numeric datatype
MLR_MonthlyExpense["Debt"] = MLR_MonthlyExpense["Debt"].apply(lambda x: int(x.replace('$','').replace(',',''))) 


# In[101]:


# Check for correlation among X variables
X.corr()


# None of the variables seem to be strongly correlated

# In[114]:


## getting column names
x_columns = ["Family Size", "Sector No", "Rent", "Own", "Income", "Utilities", "Debt"]
y = MLR_MonthlyExpense["Monthly Payment"]


# In[115]:


def get_stats():
    x = MLR_MonthlyExpense[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary())


# In[116]:


get_stats()


# If we choose alpha to be 0.05, coefficients having a p-value of 0.05 or less would be statistically significant.
# In other words, we would generally want to drop variables with a p-value greater than 0.05. Hence we would drop "Utilities" with p-value of 0.915

# In[108]:


x_columns.remove("Utilities")
get_stats()


# Finally, we find that there are 6 variables left, namely Family Size, Sector No, Rent, Own, Income, and Debt.
# Since each of the p-values are below 0.05, all of these variables are said to be statistically significant.

# In[110]:


y.head()


# In[119]:


# choose a Significance level of .05 and select the predictor with lowest p value - provided its p value is less than .05

x_opt = x # initializing the dataframe containing optional/trial variable(s)
pvalue_df = pd.DataFrame() # Blank dataframe to store p values
for i in x:
    x_opt=x[i]
    pvalue_df=pvalue_df.append(pd.DataFrame(zip([i],[float(results.pvalues)]),columns =['Predictor', 'pvalue']))
    print("extracted pvalue for {} is {}".format(i,float(results.pvalues)))
    print('adj R Sq. is',results.rsquared_adj)
    print('----------------------------------------------')


# ### Q2. MLR Feature Selection – Box Office Revenue Prediction
# An industry analyst is interested in building a predictive model to understand the impact
# of various factors and opening week revenue numbers in the overall collections of a
# movie (Total revenue).

# In[127]:


MLR_MovieBoxOffice = pd.read_csv("C:\\Users\\theco\\MLR_MovieBoxOffice_data.csv")


# In[128]:


MLR_MovieBoxOffice.head()


# In[129]:


# split the dataframe into dependent and independent variables.
x = MLR_MovieBoxOffice[['movie_name', 'revenue_opening_day', 'revenue_opening_weekend', 'revenue_firstweek','movie_genre', 'runtime', 'movie_director', 'release_month', 'release_year']]
y = MLR_MovieBoxOffice['revenue_total']
x.head()


# In[130]:


y.head()


# In[138]:


# since the state is a string datatype column we need to encode it.
x = pd.get_dummies(x,drop_first=True)
x.head()


# In[143]:


x.corr()


# In[144]:


# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.Series([variance_inflation_factor(x.values, i) 
               for i in range(x.shape[1])], 
               index=x.columns)


# ### Q3. MLR – Feature Selection – Building Energy Efficiency
# A study looked into assessing the heating load and cooling load requirements of
# buildings (that is, energy efficiency) as a function of building parameters. We perform
# energy analysis using 12 different building shapes. The dataset comprises 768 samples
# and 8 features, aiming to predict two real valued responses (heating load and cooling
# load).

# In[150]:


MLR_BuildingEfficiency = pd.read_csv("C:\\Users\\theco\\MLR_BuildingEfficiency.csv")


# In[151]:


MLR_BuildingEfficiency.head()


# In[155]:


X = MLR_BuildingEfficiency.drop(["Overall_Height", "Glazing_Area","Glazing_Area_Distribution", "Heating_Load", "Cooling_Load"], axis=1)


# In[156]:


X.head()


# In[157]:


X.corr()


# In[158]:


# Fit an OLS model
Y = MLR_BuildingEfficiency["Heating_Load"]
X1 = sm.add_constant(X)

model1 = sm.OLS(Y, X1).fit()
print(model1.summary())


# In[ ]:


The Relative_Compactness, Surface_Area, Wall_Area, and Roof_Area have p-values less than 0.05. Hence, these factors have
significant effect on the Heating_Load.


# In[159]:


# Fit an OLS model
Y = MLR_BuildingEfficiency["Cooling_Load"]
X1 = sm.add_constant(X)

model1 = sm.OLS(Y, X1).fit()
print(model1.summary())


# In[ ]:


The Relative_Compactness, Surface_Area, Wall_Area, and Roof_Area have p-values less than 0.05. Hence, these factors have
significant effect on the Cooling_Load.

