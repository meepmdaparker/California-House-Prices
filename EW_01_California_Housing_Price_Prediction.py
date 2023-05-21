#!/usr/bin/env python
# coding: utf-8

# # **10/24**

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Imports" data-toc-modified-id="Imports-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Data-Source" data-toc-modified-id="Data-Source-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Data Source</a></span></li><li><span><a href="#Data-Set" data-toc-modified-id="Data-Set-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Set</a></span></li><li><span><a href="#Background-and-Goals" data-toc-modified-id="Background-and-Goals-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Background and Goals</a></span><ul class="toc-item"><li><span><a href="#Read-In-the-Data" data-toc-modified-id="Read-In-the-Data-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Read-In the Data</a></span></li><li><span><a href="#Take-a-Quick-Look-at-the-Data-Structure" data-toc-modified-id="Take-a-Quick-Look-at-the-Data-Structure-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Take a Quick Look at the Data Structure</a></span></li><li><span><a href="#What-Do-We-Notice?" data-toc-modified-id="What-Do-We-Notice?-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>What Do We Notice?</a></span></li><li><span><a href="#Recall-Goal" data-toc-modified-id="Recall-Goal-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Recall Goal</a></span></li></ul></li><li><span><a href="#Measuring-Performance" data-toc-modified-id="Measuring-Performance-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Measuring Performance</a></span></li><li><span><a href="#Split-the-Data" data-toc-modified-id="Split-the-Data-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Split the Data</a></span><ul class="toc-item"><li><span><a href="#Splitting-via-Random-Sampling" data-toc-modified-id="Splitting-via-Random-Sampling-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Splitting via Random Sampling</a></span></li><li><span><a href="#Splitting-via-Stratified-Random-Sampling" data-toc-modified-id="Splitting-via-Stratified-Random-Sampling-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Splitting via Stratified Random Sampling</a></span></li></ul></li><li><span><a href="#Exploratory-Data-Analysis-to-Gain-Insights" data-toc-modified-id="Exploratory-Data-Analysis-to-Gain-Insights-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Exploratory Data Analysis to Gain Insights</a></span><ul class="toc-item"><li><span><a href="#Visualizing-Geographical-Data" data-toc-modified-id="Visualizing-Geographical-Data-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Visualizing Geographical Data</a></span></li><li><span><a href="#Measuring-Relations-Through-Correlations" data-toc-modified-id="Measuring-Relations-Through-Correlations-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>Measuring Relations Through Correlations</a></span><ul class="toc-item"><li><span><a href="#Sanity-Check---What-have-we-done-thus-far?" data-toc-modified-id="Sanity-Check---What-have-we-done-thus-far?-7.2.1"><span class="toc-item-num">7.2.1&nbsp;&nbsp;</span>Sanity Check - What have we done thus far?</a></span></li></ul></li><li><span><a href="#Feature-Engineering---Experimenting-with-Predictor-Combinations" data-toc-modified-id="Feature-Engineering---Experimenting-with-Predictor-Combinations-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>Feature Engineering - Experimenting with Predictor Combinations</a></span></li></ul></li><li><span><a href="#Prepare-the-Data-for-Predictive-Models" data-toc-modified-id="Prepare-the-Data-for-Predictive-Models-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Prepare the Data for Predictive Models</a></span><ul class="toc-item"><li><span><a href="#Data-Cleaning" data-toc-modified-id="Data-Cleaning-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Data Cleaning</a></span></li><li><span><a href="#Handling-Text-and-Categorical-Predictors" data-toc-modified-id="Handling-Text-and-Categorical-Predictors-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Handling Text and Categorical Predictors</a></span></li><li><span><a href="#Predictor-Scaling" data-toc-modified-id="Predictor-Scaling-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Predictor Scaling</a></span></li><li><span><a href="#Some-Reminders" data-toc-modified-id="Some-Reminders-8.4"><span class="toc-item-num">8.4&nbsp;&nbsp;</span>Some Reminders</a></span></li><li><span><a href="#Putting-It-All-Together" data-toc-modified-id="Putting-It-All-Together-8.5"><span class="toc-item-num">8.5&nbsp;&nbsp;</span>Putting It All Together</a></span></li></ul></li><li><span><a href="#Select-and-Train-a-Model" data-toc-modified-id="Select-and-Train-a-Model-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Select and Train a Model</a></span><ul class="toc-item"><li><span><a href="#Recall:-In-Sample,-Out-of-Sample" data-toc-modified-id="Recall:-In-Sample,-Out-of-Sample-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Recall: In-Sample, Out-of-Sample</a></span><ul class="toc-item"><li><span><a href="#Model-Selection" data-toc-modified-id="Model-Selection-9.1.1"><span class="toc-item-num">9.1.1&nbsp;&nbsp;</span>Model-Selection</a></span></li><li><span><a href="#Hyperparameter-Selection" data-toc-modified-id="Hyperparameter-Selection-9.1.2"><span class="toc-item-num">9.1.2&nbsp;&nbsp;</span>Hyperparameter-Selection</a></span></li></ul></li><li><span><a href="#Estimating-and-Evaluating-In-Sample-To-Find-A-Model-Class" data-toc-modified-id="Estimating-and-Evaluating-In-Sample-To-Find-A-Model-Class-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Estimating and Evaluating In-Sample To Find A Model Class</a></span></li><li><span><a href="#Better-Evaluation-Using-Resampling-Techniques-To-Find-A-Model-Class" data-toc-modified-id="Better-Evaluation-Using-Resampling-Techniques-To-Find-A-Model-Class-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Better Evaluation Using Resampling Techniques To Find A Model Class</a></span><ul class="toc-item"><li><span><a href="#Cross-Validation" data-toc-modified-id="Cross-Validation-9.3.1"><span class="toc-item-num">9.3.1&nbsp;&nbsp;</span>Cross-Validation</a></span></li><li><span><a href="#K-Fold-Cross-Validation" data-toc-modified-id="K-Fold-Cross-Validation-9.3.2"><span class="toc-item-num">9.3.2&nbsp;&nbsp;</span>K-Fold Cross-Validation</a></span></li><li><span><a href="#Cross-Validation-on-In-Sample-Data-To-Find-A-Model-Class" data-toc-modified-id="Cross-Validation-on-In-Sample-Data-To-Find-A-Model-Class-9.3.3"><span class="toc-item-num">9.3.3&nbsp;&nbsp;</span>Cross-Validation on In-Sample Data To Find A Model Class</a></span></li></ul></li></ul></li><li><span><a href="#Fine-Tuning-Your-Model" data-toc-modified-id="Fine-Tuning-Your-Model-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Fine-Tuning Your Model</a></span><ul class="toc-item"><li><span><a href="#Grid-Search" data-toc-modified-id="Grid-Search-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Grid Search</a></span></li><li><span><a href="#Randomized-Search" data-toc-modified-id="Randomized-Search-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Randomized Search</a></span></li><li><span><a href="#Ensemble-Methods" data-toc-modified-id="Ensemble-Methods-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Ensemble Methods</a></span></li><li><span><a href="#Analyze-the-Best-Models-and-Their-Errors" data-toc-modified-id="Analyze-the-Best-Models-and-Their-Errors-10.4"><span class="toc-item-num">10.4&nbsp;&nbsp;</span>Analyze the Best Models and Their Errors</a></span></li><li><span><a href="#Evaluate-Your-Model-Out-of-Sample" data-toc-modified-id="Evaluate-Your-Model-Out-of-Sample-10.5"><span class="toc-item-num">10.5&nbsp;&nbsp;</span>Evaluate Your Model Out-of-Sample</a></span></li></ul></li><li><span><a href="#In-Summary" data-toc-modified-id="In-Summary-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>In Summary</a></span></li><li><span><a href="#Describe-the-Problem-and-Solution-As-If-You-Were-Talking-to-a-Hiring-Manager---Homework-Exercise" data-toc-modified-id="Describe-the-Problem-and-Solution-As-If-You-Were-Talking-to-a-Hiring-Manager---Homework-Exercise-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Describe the Problem and Solution As If You Were Talking to a Hiring Manager - Homework Exercise</a></span></li><li><span><a href="#Miscellaneous:-NOT-NECESSARY-TO-KNOW" data-toc-modified-id="Miscellaneous:-NOT-NECESSARY-TO-KNOW-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Miscellaneous: NOT NECESSARY TO KNOW</a></span><ul class="toc-item"><li><span><a href="#Another-Experiment" data-toc-modified-id="Another-Experiment-13.1"><span class="toc-item-num">13.1&nbsp;&nbsp;</span><a href="https://www.sciencedirect.com/science/article/pii/S016771529600140X" rel="nofollow" target="_blank">Another Experiment</a></a></span></li><li><span><a href="#Launch-Your-System" data-toc-modified-id="Launch-Your-System-13.2"><span class="toc-item-num">13.2&nbsp;&nbsp;</span>Launch Your System</a></span></li><li><span><a href="#Monitor-Your-System" data-toc-modified-id="Monitor-Your-System-13.3"><span class="toc-item-num">13.3&nbsp;&nbsp;</span>Monitor Your System</a></span></li><li><span><a href="#Maintain-Your-System" data-toc-modified-id="Maintain-Your-System-13.4"><span class="toc-item-num">13.4&nbsp;&nbsp;</span>Maintain Your System</a></span></li></ul></li></ul></div>

# <hr style="border: 20px solid black">

# <img src="https://miro.medium.com/max/1400/1*QV1rVgh3bfaMbtxueS-cgA.png">

# <hr style="border: 20px solid black">

# <h1>Imports</h1>

# In[1]:


import os 
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

##############################################
##############################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# <hr style="border: 20px solid black">

# <h1>Data Source</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>The data set we'll use is from the <b>1990 California State Census</b> and is preserved on the <a href="http://lib.stat.cmu.edu/datasets/">Statlib repository</a>.</li>
#         <br>
#         <li>In most business settings, your data would be available in a relational database (or some common data store) and spread across multiple tables, documents, or files.</li>
#         <br>
#         <li>To access your data, you would first need to get your credentials and access authorizations while familiarizing yourself with the data schema.</li>
#         <br>
#         <li>However, in this simplified project, things are much simpler.</li>
#         <br>
#         <li>All you have to do is to download a single compressed file (<a href="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"><i>housing.tgz</i></a>) which contains a CSV file called <i>housing.csv</i> with all of the data in a single file.</li>
#         <br>
#         <li style="color:orange">That is, we won't have to do any extracting of the data from its raw source(s) to an analyzable data set for prediction.</li>
#         <br>
#     </ul>
# </font>

# In[2]:


#print(os.getcwd())


# In[3]:


#os.listdir()


# <hr style="border: 20px solid black">

# <h1>Data Set</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>This data set consists of <i>spatial</i> (<i>cross-sectional</i>) data containing 20,640 observations of housing prices with nine covariates.</li>
#         <br>
#         <li>The data originally appeared in Pace and Barry (1997), "Sparse Spatial Autoregressions", Statistics and Probability Letters.</li>
#         <br>
#         <li>The data set is outdated, but it serves a purpose for illustration of the basic machine learning scaffolding in a regression problem.</li>
#         <br>
#         <li style="color:orange">We should remark that we are being very messy with our directory structures, which is not good in the long run. For the short run, it will suffice.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>Background and Goals</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>The data includes metrics for each block group in California such as</li>
#         <br>
#         <ul>
#             <li>population,</li>
#             <br>
#             <li>median income,</li>
#             <br>
#             <li>and median housing price.</li>
#         <br>
#         </ul>
#         <li><b>Block groups</b> are the smallest geographical unit for which the US Census Bureau publishes sample data.</li>
#         <br>
#         <li>A block group typically has 600 to 3,000 people.</li>
#         <br>
#         <li>We will rename block groups to <i>districts</i>.</li>
#         <br>
#         <li style="color:blue">Your model should use this data and be able to <b>predict the <i>median housing price</i> in any district</b>.</li>
#         <br>
#         <li style="color:orange">It is important to understand the business use of your predictive model at all stages of the development and production lifecycle. </li>
#         <br>
#         <ul style="color:orange">
#             <li>That is, how is your model getting used in the business?</li>
#             <br>
#             <li>This involves extensive communication with the team who takes ownership of your developed model.</li>
#         </ul>
#         <br>
#     </ul>
# </font>

# <h2>Read-In the Data</h2>
# 

# In[ ]:


# housing = pd.read_csv('housing.csv')


# <h2>Take a Quick Look at the Data Structure</h2>
# 

# In[5]:


housing.head()


# In[6]:


print('The number of observations n = {}, while the number of predictor variables p = {}'     .format(housing.shape[0],             housing.shape[1]-1))
#housing.shape[0] is number of rows/entries 
#what is [1]-1 -> technically ten predictor variables, but we're combining longitude and latitude


# <font size="+1">
#     <ul>
#         <li>This is a relatively small data set for modern machine learning applications.</li>
#         <br>
#     </ul>
# </font>

# In[7]:


# The info() method is useful to get a quick description of the data

housing.info()


# In[8]:


# Are there null values?

housing.isna().sum()  
# is the null bedrooms because it was a studio or is there missing data?
#.dropna
#.fillna
#whatever method you end up using, you need to think about why it is missing to begin with


# <font size="+1">
#     <ul>
#         <li>What type of variables are the covariates?</li>
#         <br>
#         <ul>
#             <li>numerical,</li>
#             <br>
#             <li>categorical,</li>
#             <br>
#             <li>discrete,</li>
#             <br>
#             <li>etc.</li>
#             <br>
#         </ul>
#     </ul>
# </font>

# In[9]:


housing.dtypes


# In[10]:


# What are the unique categories?

housing.loc[:,'ocean_proximity'].unique()


# In[11]:


# How many districts belong to each category?

housing.loc[:,'ocean_proximity'].value_counts()


# In[12]:


housing.describe()


# In[13]:


# To get a quick statistical summary of the numerical variables
# Is there anything to notice?
# Remember, the data is per district of homes, not per household

housing.describe().T
#.T transposes
#the mean of the median_house_value is 206855.816909
#this data is by district


# In[14]:


# It would be a good idea to do more EDA grouping by longitude and latitude, i.e. per district.
housing.groupby(['longitude', 'latitude'])[['total_rooms', 'total_bedrooms', 'population']].mean()


# In[15]:


# To get a quick visual summary of the variability of numerical variables

housing.hist(bins=50, figsize=(16,8));
#plt.show()


# In[16]:


#what kind of observations do we see
#median income is not expressed in USD (x variable are 0,2,4, etc)

#housing median age and median house value are capped (as seen by the sudden spike at 50 and 500,000 respectively) 
#aka we don't know information beyond those two points, how much we have of those values beyond those two points
#this means that the data is censored, or capped (we don't have info beyond the points)
#there are algorithims that work with data that is censored however (we won't learn that now)


# <hr style="border: 20px solid black">

# <h2>What Do We Notice?</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li>The data is expressed per district, not per household!</li>
#         <br>
#         <li>The median income variable doesn't look like it is expressed in US dollars (USD).</li>
#         <br>
#         <ul>
#             <li>You check with the team that colleted the data and you're told the data has been scaled and capped at 15 for higher median incomes and 0.5 for lower median incomes.</li>
#             <br>
#             <li>The numbers roughly represent tens of thousands of dollars.</li>
#             <br>
#             <li><b style="color:red">It is common to work with pre-processed variables and it is good practice to try to understand how the variable was computed</b>.</li>
#             <br>
#         </ul>
#         <li>The housing median age and median house value are also capped.</li>
#         <br>
#         <ul>
#             <li>This could present a serious problem since your target variable is median house value.</li>
#             <br>
#             <li>This could cause your predictive model to never predict prices that go beyond that limit, which could cause serious <b>bias</b> in your predictions.</li>
#             <br>
#             <li>You check with your client team (the team that will use your model's output) to see if this is a problem.</li>
#             <br>
#             <li>If they tell you they need precise predictions beyond $\$500,000$, then you can collect (or contact the data preparation team to collect) proper labels for the districts whose prices were capped, or remove those districts from the data set.</li>
#             <br>
#         </ul>
#         <li>The variables have very different scales.</li>
#         <br>
#         <li>Many histograms are <i>heavy-tailed</i> and <i>skewed</i>, that is, they extend much farther to the right of the median than to the left. This can, sometimes, make it harder for some machine learning algorithms to detect patterns.</li>
#         <br>
#     </ol>
# </font>

# <h2>Recall Goal</h2>
# <br>
# <font size="+1">
#     <ul>
#         <br>
#         <li style="color:red">The model we develop should use the data to predict the <i>median housing price</i> in any district.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>Measuring Performance</h1>
# <br>
# <br>
# 
# <img src="https://jmlb.github.io/images/20180701/img_02.png" width="500" height="500">
# 
# <br>
# <font size="+1">
#     <ul>
#         <li>A typical performance measure for regression problems is the root mean square error (RMSE). </li>
#         <br>
#         <ul>
#             <li>This error function gives a higher weight for large errors.</li>
#             <br>
#             <li>The error function depends on the data and the prediction model
#                 <br>
#                 <br>
#                  $$
#         \text{RMSE} \ (\mathbf{X}, f) := \sqrt{\underbrace{\frac{1}{n}\sum_{i=1}^n \underbrace{\left(y_i - f_{\hat{\lambda}}(\mathbf{x}_i; \hat{\beta})\right)^2}_{error-squared}}_{average}}
#         $$</li>
#         </ul>
#             <br>
#         <br>
#         <li>If there were <b>many outlier</b> districts in the data you might want to <b>use the mean absolute error (MAE)</b> as it focuses less on large errors relative to RMSE. </li>
#         <br>
#         <ul>
#             <li>The error function depends on the data and the prediction model
#             <br>
#             <br>
#         $$
#         \text{MAE} \ (\mathbf{X}, f) := \underbrace{\frac{1}{n}\sum_{i=1}^n \underbrace{\left|y_i - f_{\hat{\lambda}}(\mathbf{x}_i; \hat{\beta})\right|}_{absoulte-error}}_{average}
#         $$</li>
#             <br>
#         </ul>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>Split the Data</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>Subset the data to create a training (in-sample) data set and a testing (out-of-sample) data set.</li>
#         <br>
#         <li>We shouldn't do more exploration until we split our data to reduce <i style="color:red">data snooping bias</i>, which will cause our estimate of generalization error to be lower than it should be.</li>
#         <br>
#         <ul>
#         <li>This is because our brains are highly prone to overfitting and can bias our experiments by thinking we've discovered some pattern that doesn't generalize well.</li>
#         <br>
#         </ul>
#     </ul>
# </font>

# <img src="https://i.ytimg.com/vi/xgDs0scjuuQ/maxresdefault.jpg">
# <br>
# <font size="+1">
#     <ul>
#         <li>There are different ways to create an out-of-sample test set, and what is best depends on the <b>dependency structure</b> of your data, as well as on the <b>homogeneity</b> of your sample (i.e. data is sampled from a single population).</li>
#         <br>
#         <ul>
#             <br>
#             <img src="https://kanoki.org/images/2020/04/image-34.png">
#             <br>
#             <br>
#             <br>
#             <br>
#             <img src="https://upload.wikimedia.org/wikipedia/commons/9/9b/VIX.png" width="600" height="600">
#             <br>
#             <li style="color:dodgerblue">For example, if your data is a <b>time-series</b> where there is a relation from one point in time to another (<i>autocorrelation or serial correlation</i>), then the best way to split the data is in the order of time.</li>
#             <br>
#             <li style="color:dodgerblue">That is, only use observations that have occurred in the past to predict future values.</li>
#             <br>
#             <li style="color:dodgerblue">Failure to account for temporal dependence could result in a biased estimate of generalization error caused by a <i style="color:red">look-ahead (leakage) bias</i>.</li>
#             <br>
#             <li style="color:dodgerblue">This could also be caused by using variables that are unknown at the time of your prediction and are only made available after your prediction needs to be made.</li>
#             <br>
#             <br>
#             <br>
#             <br>
#             <br>
#             <img src="https://www.mathworks.com/help/examples/matlab/win64/Visualizing4DExample_01.png">
#             <br>
#             <br>
#             <br>
#             <br>
#             <br>
#             <br>
#             <img src="https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/asset/d020d386-0c56-4cf7-bf2f-77bdf842a171/gr1.jpg">
#             <br>
#             <br>
#             <li style="color:purple">Another example is if your data is <b>cross-sectional</b> and <b>heterogeneous</b> with subgroups of the data being very similar (homogeneous) as measured by some similarity measure (such as correlations), then the best way to split the data is by using stratified sampling techniques (as opposed to random sampling).</li>
#             <br>
#             <li style="color:purple">Stratified sampling is when the population is divided into similar (homogeneous) subgroups called <i>strata</i> and a representative number of observations are taken (sampled) from each stratum to guarantee the test set looks like (is representative of) the overall population.</li>
#             <br>
#             <li style="color:purple">In other words, you group by <i>similar</i> covariates.</li>
#             <br>
#             <li style="color:purple">With purely random sampling, there is a chance (which can be large) of sampling a skewed test set.</li>
#             <br>
#             <li style="color:purple">Failure to account for subgroup homogeneity or cross-sectional dependence could result in a biased estimate of generalization error caused by a <i style="color:red">sampling bias</i>.</li>
#             <br>
#         </ul>
#         <li>For now, we won't worry too much about these issues, but they are good to be aware of.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Splitting via Random Sampling</h2>

# In[17]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#test size is 20% of data
#random_state is kinda like an ID, all random_state = 42 data results will be the same 


# <img src="https://www.geopoll.com/wp-content/uploads/2020/06/simple-random-sample.jpg" height="300" width="500">

# <hr style="border: 20px solid black">

# <h2>Splitting via Stratified Random Sampling</h2>
# <br>
# <br>
# <img src="https://www.baeldung.com/wp-content/uploads/sites/4/2021/12/figure-2048x413-1-1024x207.png">

# <font size="+1">
#     <ul>
#         <li>Suppose you consult with the <b>Residential Real Estate Economics</b> team who told you that the median income is a very important predictor of median housing prices.</li>
#         <br>
#         <li>If you decide you want to ensure that the test set is representative of the various categories of incomes in the whole data set, you may want to use stratified random sampling.</li>
#         <br>
#         <li style="color:dodgerblue">Since the median income is a continuous numerical value, you first need to create an discrete income category variable.</li>
#         <br>
#         <li style="color:dodgerblue">It is important to have a sufficient number of observations in your data set for each stratum, otherwise the estimate of a stratum's importance may be biased.</li>
#         <br>
#         <ul>
#             <li style="color:dodgerblue">This implies you shouldn't have too many strata, and each stratum should be large enough.</li>
#         <br>
#         </ul>
#     </ul>
# </font>

# In[18]:


print(housing.loc[:,'median_income'].describe())
housing.loc[:,'median_income'].hist(bins=50, figsize=(16,8))

#our goal is now to create strata, to create 5 bins of discrete categories 


# In[19]:


# We use pd.cut() to bin values into (5) discrete categories labeled 1 to 5, which we'll later sample from.

#create a new column called income category based off median income
#np.inf means that everything beyond 6 will be in the last bin

#\ change lines in code
housing.loc[:,'income_category'] = pd.cut(housing.loc[:,'median_income'],
      bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
      labels=range(1,6))
#0, 1.5, 3.0, 4.5, 6.0

#can also use clustering to make distinct bins?


# In[20]:


housing.loc[:,'income_category'].head(10)


# In[21]:


housing.loc[:,'income_category'].hist(figsize=(16,8))


# In[22]:


housing.loc[:,'income_category'].value_counts().sort_index().plot(kind='bar', figsize=(16,8))
#bar chart, not a histogram now


# In[23]:


# Now we can do stratified random sampling based on the income category

# Provides train/test indices to split data in train/test sets.'

# This cross-validation object is a merge of StratifiedKFold and
# ShuffleSplit, which returns stratified randomized folds. The folds
# are made by preserving the percentage of samples for each class.

#cross-validation?

from sklearn.model_selection import StratifiedShuffleSplit

# create the splitter object
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#n_splits = number of iterations
#Number of re-shuffling & splitting iterations
#only splitting once
#splitting into training and testing
#when would you want to split more than once?
#maybe splitting more than once for a more random split of training and testing


for train_index, test_index in split.split(housing, housing.loc[:,'income_category']):
    print(train_index, test_index)
    strat_train_set = housing.loc[train_index, :]
    strat_test_set = housing.loc[test_index, :]
#don't get what this is for?


# In[24]:


strat_train_set


# In[25]:


strat_test_set
#4128 rows as opposed to 16512 of the data, which is 20% of the original set of the data


# <font size="+1">
#     <ul>
#         <li>To test if stratified random sampling actually achieved similar representation of the full data set (regarding the median income variable) we can <b>look at the percentage of categories in the test set</b> and <b>compare with the full data set.</b></li>
#         <br>
#         <li>We can also compare with random sampling without stratification.</li>
#         <br>
#     </ul>
# </font>

# In[26]:


strat_test_set.loc[:,'income_category'].value_counts()
#locate all rows in the income_category column
#count how many of each income_category there are (1-5)


# In[27]:


len(strat_test_set)


# In[28]:


(strat_test_set.loc[:,'income_category'].value_counts() / len(strat_test_set)).sort_index()

#.sort_index shows us the results by descending index
#ie) 3.9% of the test set income_category are of category 1


# In[29]:


(housing.loc[:,'income_category'].value_counts() / len(housing)).sort_index()
#now compare the percentages to the actual dataset (above is the stratified testing set)
#proportions are the same
#implies that test set is distributed similarly to the entire dataset


# In[30]:


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

(test_set.loc[:,'income_category'].value_counts() / len(test_set)).sort_index()
#the test set is a bit off (w/o stratifying)


# In[31]:


strat_test_set_percent = (strat_test_set.loc[:,'income_category'].value_counts() / len(strat_test_set))                        .sort_index()                        .rename('strat_test_set')

full_data_set_percent = (housing.loc[:,'income_category'].value_counts() / len(housing))                        .sort_index()                        .rename('full_data_set')

random_test_set_percent = (test_set.loc[:,'income_category'].value_counts() / len(test_set))                        .sort_index()                        .rename('random_test_set')

percents = pd.concat([full_data_set_percent, 
           strat_test_set_percent, 
           random_test_set_percent], axis=1)

percents
#showing the distribution percentage between full data set, strat data set, and random test dataset
#because there is not a lot of difference between stratified and not stratified, we continue our random sampling without stratifying


# <font size="+1">
#     <ul>
#         <li style="color:red">It doesn't look like there is too much difference, so you decide to use random sampling without stratifying.</li>
#         <br>
#         <li>If you are't convinced that the sampling bias is negligible, it is possible to proceed using both test sets.</li>
#         <br>
#         <li>That is, run your experiments with both sampling techniques and pick the one with least favorable results as a conservative estimate of generalization error.</li>
#         <br>
#         <li>For now, we will continue with random sampling and we will remove our income category variable from our train and test sets.</li>
#         <br>
#         <li>Stratified sampling isn't always the right way to proceed, see <a href="https://en.wikipedia.org/wiki/Stratified_sampling#Disadvantages">here</a> for more on the cons of stratified sampling.</li>
#         <br>
#         <ul>
#             <li>One danger is if the data sample is not representative of the true population, then you will force the population sampling bias into the prediction model by forcing the test set distribution to match the biased data sample.</li>
#             <br>
#         </ul>
#         <li>If interested, you can see some common sampling strategies and their associated pros and cons <a href="http://www2.hawaii.edu/~cheang/Sampling%20Strategies%20and%20their%20Advantages%20and%20Disadvantages.htm">here</a>.</li>
#         <br>
#     </ul>
# </font>

# In[32]:


#strat_train_set = strat_train_set.drop('income_category', axis=1)
#strat_test_set = strat_test_set.drop('income_category', axis=1)

train_set = train_set.drop('income_category', axis=1)
test_set = test_set.drop('income_category', axis=1)

# train_set = strat_train_set.drop('income_category', axis=1)
# test_set = strat_test_set.drop('income_category', axis=1)


# <font size="+1" style="color:blue">
#     <ul>
#         <li><b>We've spent a lot of time on test set generation, but it is for a good reason!</b></li>
#         <br>
#         <li><b>Test set generation is often a neglected but critical part of a building and evaluating predictive models.</b></li>
#         <br>
#         <li><b>We will see these concepts again when we discuss cross-validation.</b></li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>Exploratory Data Analysis to Gain Insights</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>So far, we've only taken a relatively small glance at the data to get a high-level understanding of the kind of data we're manipulating. Now we'll go a little more in-depth.</li>
#         <br>
#         <li>Put the test set aside and be sure we're only exploring the training set.</li>
#         <br>
#         <li>If the training set is very large, you may want to further sample the training set and create an exploration set to make manipulations easy and fast.</li>
#         <br>
#         <li>We'll create a copy so we can manipulate the set without affecting the training set.</li>
#         <br>
#     </ul>
# </font>

# In[33]:


X_Y_train = train_set.copy()


# <h2>Visualizing Geographical Data</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>Recall, the data contains <b>geographical</b> information (latitute and longitude).</li>
#         <br>
#         <li>We'll create a <i>scatterplot</i> of all districts to visualize the data.</li>
#         <br>
#     </ul>
# </font>

# In[34]:


X_Y_train.plot(kind="scatter", x="longitude", y="latitude", figsize=(16,8))

#by plotting out longitutde and latitutde, we can see where we're looking at
#in this case the points indicate a shape similar to California 
#by seeing this, we can try clustering based on regions in California


# <br>
# <font size="+1">
#     <ul>
#         <li>As expected, the scatterplot of the sample looks like California, but it is hard to visualize any particular pattern.</li>
#         <br>
#         <li>We'll tune some settings to see if we can spot any patterns in the data.</li>
#         <br>
#         <li>Specifically, we'll emphasize the locations with a high density of data points.</li>
#         <br>
#     </ul>
# </font>

# In[35]:


#X_Y_train.plot?


# In[36]:


# 'alpha' in (0, 1) controls plot transparency

X_Y_train.plot(kind="scatter", x="longitude", y="latitude", 
               alpha=0.1, 
               figsize=(16,8))
#alpha relates to opacity of the points
#tells us the density of the data
#the darker regions show us where most of our data is coming from 


# <br>
# <font size="+1">
#     <ul>
#         <li>Now we can clearly see the high-density areas.</li>
#         <br>
#         <ul>
#             <li>In particular, the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley areas around Sacramento and Fresno.</li>
#         <br>
#         </ul>
#         <li>You may need to play around with the visualization parameters to make the patterns stand out.</li>
#         <br>
#         <li>Now we'll consider how to include housing prices in the geographical data.</li>
#         <br>
#     </ul>
# </font>

# In[37]:


# 'c' is for color values, 's' is for marker size
# The radius of each circle represents the district's population ('s'), and the color represents the price ('c')
# We use a pre-defined colormap ('cmap') called 'jet' which ranges from blue (low prices) to red (high prices)

X_Y_train.plot(kind='scatter', x='longitude', y='latitude', 
               alpha=0.4,
               s=X_Y_train.loc[:,'population']/100,
               label='population', 
               figsize=(16,8),
               c='median_house_value', 
               cmap=plt.get_cmap('jet'), 
               colorbar=True) #colorbar=False

#s=X_Y_train.loc[:,'population']/100 
#s is the size parameter for size of the circle
#size should be based off the population parameter 

#cmap is color map


#we identify two different things
#population size and median_house_value
plt.legend();


# <br>
# <font size="+1">
#     <ul>
#         <li>This plot tells you that the housing prices are very much related to the location, for example, close to the ocean, and to the population density.</li>
#         <br>
#         <li>This isn't particularly surprising. </li>
#         <br>
#         <li style="color:blue">TODO: A clustering algorithm should be useful for detecting the main cluster, and for adding new features that measure the proximity to the cluster centers.</li>
#         <br>
#     </ul>
# </font>

# <h2>Measuring Relations Through Correlations</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>Since the data set isn't too large, we can easily compute the <i>standard correlation coefficient</i> (also called <i>Pearson's r</i>) between every pair of predictor variables.</li>
#         <br>
#         <li>As one possible technique for measuring how related a predictor variable is to the target variable, we can compute the correlation between $X_i$ and $Y$, for all $1\leq i \leq p$.</li>
#         <br>
#     </ul>
# </font>

# In[38]:


corr_matrix = X_Y_train.corr()


# In[39]:


corr_matrix
#identifying the correlation between variables


# In[40]:


# To measure the linear relation between the target variable and all covariates, we only need to 
# look at one column of the correlation matrix

corr_matrix.loc[:,'median_house_value'].sort_values(ascending=False)
#all rows of median_house_value sort by descending order
#measuring linear relationship between variables
#but not always linearly correlated

#median income of that district has the highest corelation to median_house_value


# <br>
# <font size="+1">
#     <ul>
#         <li>The correlation coefficient measures the <b>linear</b> relationship between two variables.</li>
#         <br>
#         <li>The correlation coefficient ranges from -1 to +1.</li>
#         <br>
#         <li>When it is close to one, it means that there is a strong positive linear relation; for example, the median house value tends to go up when the median income goes up.</li>
#         <br>
#         <li>When it is close to negative one, it means that there is a strong negative linear relation; for example, prices have a slight tendency to go down when you go north.</li>
#         <br>
#         <li>Finally, a correlation of zero means there is no linear relation; though there can still be a nonlinear relation.</li>
#     </ul>
# </font>

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/1920px-Correlation_examples2.svg.png" height="800" width="800">

# <br>
# <font size="+1">
#     <ul>
#         <li>We can also check relations between variables through a scatter plot matrix.</li>
#         <br>
#     </ul>
# </font>

# In[41]:


from pandas.plotting import scatter_matrix

scatter_matrix(X_Y_train, figsize=(16,8))
#plotting a 121 different scatter plots
#scatter matrix is a matrix of scatter plots 


# <br>
# <font size="+1">
#     <ul>
#         <li>Because we have $11$ numerical variables, this matrix has $11^2 = 121$ plots.</li>
#         <br>
#         <li>This isn't very visually appealing, so let's just focus on a few promising variables.</li>
#     </ul>
# </font>

# In[42]:


interesting_variables = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(X_Y_train.loc[:, interesting_variables], figsize=(16,8));
#most interesting one is the median income to median house value as we see a positively correlated trend in the scatter plot


# <br>
# <font size="+1">
#     <ul>
#         <li>It appears the most promising variable to predict the median house value is the median income, so let's zoom-in on this relationship.</li>
#         <br>
#     </ul>
# </font>

# In[43]:


X_Y_train.plot(kind="scatter", x="median_income", y="median_house_value",
               figsize=(16,8),
               alpha=0.1)


# <br>
# <font size="+1">
#     <ul>
#         <li>This plot reveals a few things.</li>
#         <br>
#         <li>First, the correlation is fairly strong as you can clearly see the upward trend and the points aren't too dispersed.</li>
#         <br>
#         <li>Second, the price cap that we noticed earlier is clearly visible at 500,000.</li>
#         <br>
#         <li>Finally, this plot reveals other less obvious straight lines: a horizontal line around 450,000, another one around 350,000, and perhaps one around 280,000, as well as a few more below that.</li>
#         <br>
#         <li style="color:blue">TODO: It might be a good idea to try removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks. <b>Q: </b> Is there a rational explanation for these data quirks?</li>
#         <br>
#     </ul>
# </font>

# <h3>Sanity Check - What have we done thus far?</h3>
# <br>
# <font size="+1" style="color:purple">
#     <ul>
#         <li>So far, we have investigated a few ways we can explore the data and gain some insights. </li>
#         <br>
#         <li>We found a few data irregularities that might need to be cleaned up before feeding the data to a predictive algorithm.</li>
#         <br>
#         <li>We found interesting correlations between predictor variables, as well as with the target variable.</li>
#         <br>
#         <li>We have also noticed some variables have a heavy-tailed distribution, so we might want to transform them by computing their logarithm.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Feature Engineering - Experimenting with Predictor Combinations</h2>
# <br>
# <img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2021/05/64590automated-feature-engineering.png" width="500" height="500">
# <br>
# <font size="+1">
#     <ul>
#         <li>One last thing you might want to do before preparing the data for predictive models is to try to build other predictor variables that are combinations of the existing variables.</li>
#         <br>
#         <li>For example, the total number of rooms in a district isn't too helpful if you don't know how many households there are. You really want the number of rooms per household.</li>
#         <br>
#         <li>Similarly, the total number of bedrooms by itself isn't very useful, and you would probably like to compare it to the number of rooms.</li>
#         <br>
#         <li>The population per household also seems like an interesting combination to look at.</li>
#         <br>
#         <li style="color:red">At this stage, it is encouraged to let your imagination go wild in order to create more informative predictor variables.</li>
#         <br>
#     </ul>
# </font>

# In[44]:


# Creating the new variables

X_Y_train['rooms_per_household'] = X_Y_train['total_rooms'] / X_Y_train['households']

X_Y_train['bedrooms_per_room'] = X_Y_train['total_bedrooms'] / X_Y_train['total_rooms']

X_Y_train['population_per_household'] = X_Y_train['population'] / X_Y_train['households']


# In[45]:


X_Y_train.corr()['median_house_value'].sort_values(ascending=False)


# <br>
# <font size="+1">
#     <ul>
#         <li>It looks like we've added some signal!</li>
#         <br>
#         <li>The new variable, <i>bedrooms_per_room</i>, is much more correlated with the target variable than the variables used to construct it.</li>
#         <br>
#         <li>Apparently rooms with a lower bedroom to room ratio tend to be more expensive. </li>
#         <br>
#         <li>The number of rooms per household also has more signal than the total number of rooms in a district. This makes sense as the larger the house, the more expensive they are.</li>
#         <br>
#         <li style="color:red">This is an iterative process and a good first start to get a prototype model up and running. Once you get a prototype running, you can analyze its output to gain more insights and come back to the exploration step.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>Prepare the Data for Predictive Models</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>Let's first revert to a clean training set.</li>
#         <br>
#         <li>We'll also separate the predictors and the labels, since we don't necessarily want to apply the same transformations to the predictors and the target values.</li>
#         <br>
#     </ul>
# </font>

# In[46]:


X_train = train_set.drop('median_house_value', axis=1)  # drop already creates a copy

Y_train = train_set.loc[:,'median_house_value'].copy()
#median house value is our predictor variable


# <h2>Data Cleaning</h2>
# <br>
# <img src="https://cdn.substack.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2F90a01dc3-ce77-4923-bea6-d19caccf44d3_2501x1365.png" width="500" height="500">
# <br>
# <font size="+1">
#     <ul>
#         <li>Most ML models can't work with missing features, so we'll create a few functions to take care of them.</li>
#         <br>
#         <li>Recall, we had some nulls earlier, so we'll fix this with one of three options:</li>
#         <br>
#         <ol>
#         <li>Get rid of the observations with a missing covariate (rows).</li>
#         <br>
#         <li>Get rid of the whole variable (columns).</li>
#         <br>
#             <li>Set the values to some value (zero, median, mean, etc.).</li>
#             <br>
#         </ol>
#         <li>These can easily be accomplished with Pandas' methods.</li>
#         <br>
#     </ul>
# </font>

# In[47]:


X_train.dropna(subset=['total_bedrooms'])  # Option 1, dropping missing values in total bedrooms

X_train.drop('total_bedrooms', axis=1)  # Option 2, dropping total bedroosm column

median_train_set = X_train.loc[:,'total_bedrooms'].median()
 


# <br>
# <font size="+1">
#     <ul>
#         <li>If you choose to impute the median value on the missing observations, it is advised you save the value to impute on the test set as well as in production to prevent data leakage.</li>
#         <br>
#     </ul>
#     <br>
#     <br>
#     <img src="https://cdn.corporatefinanceinstitute.com/assets/look-ahead-bias.png" width="400" height="300">
# </font>

# <h2>Handling Text and Categorical Predictors</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>Let's consider how to deal with the variable <i>ocean_proximity</i>, which is a categorical, rather than numerical, variable.</li>
#         <br>
#     </ul>
# </font>

# In[48]:


#https://leochoi146.medium.com/how-and-when-to-use-ordinal-encoder-d8b0ef90c28c
#Ordinal data is similar to nominal data in that they are both are categorical, 
#except ordinal data types have an added element of order to them. 
#The exact difference or distance between the categories in ordinal data is unknown and/or cannot be measured. 

#An example would be the different answer choices in a satisfaction survey 
#(very dissatisfied, dissatisfied, satisfied, very satisfied) 
#or language proficiency (beginner, intermediate, expert). 
#As previously mentioned, the difference in satisfaction between each category can’t be quantified or put into numbers, 
#but there is a clear progression or hierarchy of the data.

#When not to use ordinal encoder
#If you can already take the mean, median, mode of your data that provides insight, 
#you’re in luck and won’t need to encode the data.
#Ordinal encoder also should not be used if your data has no meaningful order. 
#Going back to the car color example, 
#there is no way to logically order these colors from smallest to largest or worst to best. 

#When to use ordinal encoder
#ordinal encoders should be used when working with ordinal data. 
#When working with any data related to ranking something with non-numerical categories, 
#ordinal encoder is the way to go.


# In[49]:


X_train.loc[:,'ocean_proximity'].head(10)


# In[50]:


X_train.loc[:,'ocean_proximity'].unique()


# <font size="+1">
#     <ul>
#         <li>Let's convert these categories from text to numbers.</li>
#         <br>
#     </ul>
# </font>

# In[51]:


# Ordinal numbers refer to the position of things in a list.

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

ocean_proximity_encoded = ordinal_encoder.fit_transform(X_train.loc[:,'ocean_proximity'].values.reshape(-1,1))
#reshape it to one column 

ocean_proximity_encoded[:10]


# <br>
# <font size="+1">
#     <ul>
#         <li>One issue with this numerical representation is that many ML algorithms will assume that two nearby values are more similar than two distant values.</li>
#         <br>
#         <li>This may be fine in some cases such as ordered categories consisting of <i>"bad", "average", "good", "excellent"</i>.</li>
#         <br>
#         <li>Unfortunately, this doesn't work for <i>ocean_categories</i>.</li>
#         <br>
#         <li>To fix this, a common solution is to use <i>dummy (indicator) variables</i> that are one if an observation is in a particular category, and zero otherwise.</li>
#         <br>
#         <li>That is, create one binary variable per category.</li>
#         <br>
#         <li style="color:blue">In machine learning, this is called <i>one-hot encoding</i>, in statistics, this is called <i>creating dummy variables for a categorical variable</i>.</li>
#         <br>
#     </ul>
# </font>

# In[52]:


#https://albertum.medium.com/preprocessing-onehotencoder-vs-pandas-get-dummies-3de1f3d77dcc
#difference between onehotencoder and get dummies

#OHE does the same things as get dummies but in addition, OHE saves the exploded categories into it’s object.
#Saving exploded categories is extremely useful when I want to apply the same data pre-processing on my test set. 
#If the total number of unique values in a categorical column 
#is not the same for my train set vs test set, I’m going to have problems.


# In[53]:


from sklearn.preprocessing import OneHotEncoder

dummy_encoder = OneHotEncoder()

ocean_proximity_dummy = dummy_encoder.fit_transform(X_train.loc[:, 'ocean_proximity'].values.reshape(-1,1))

ocean_proximity_dummy


# In[54]:


ocean_proximity_dummy.toarray()


# <br>
# <font size="+1">
#     <ul>
#         <li>We do lose some meaning on what these columns mean, but we can easily get it back.</li>
#         <br>
#     </ul>
# </font>

# In[55]:


dummy_encoder.categories_


# In[56]:


pd.DataFrame(ocean_proximity_dummy.toarray(), columns=dummy_encoder.categories_[0])
#creating a dataframe of our encoder of ocean_proximity 
#1 being true
#ie) first row is near ocean, but not any of the other categories 


# In[57]:


pd.concat([pd.DataFrame(X_train.loc[:,'ocean_proximity']).reset_index(drop=True), 
          pd.DataFrame(ocean_proximity_dummy.toarray(), columns=dummy_encoder.categories_[0])],
         axis=1)
#concatting vertically 
#taking column ocean proximity from X-train and concatting side by side (left to right) to the dummy dataset for each
#categorical variable 


# In[58]:


#10/31


# <br>
# <font size="+1" style="color:red">
#     <ul>
#         <li>TODO: It could be a good idea to replace a categorical input with useful numerical features related to the categories; for example, you could replace the <i>ocean_proximity</i> with the distance to the ocean.</li>
#         <br>
#         <li>Alternatively, you could replace each category with an estimatable, low-dimensional vector called an <i>embedding</i>.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Predictor Scaling</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>One of the most important transformations you need to apply to your data is <i>feature scaling</i>.</li>
#         <br>
#         <li>With few exceptions, predictive models don't perform well when the input numerical variables have different scales.</li>
#         <br>
#         <li>Note that scaling the target variable is generally not required.</li>
#         <br>
#         <li>The two common ways to get all variables to have the same scale is:</li>
#         <br>
#         <ul>
#             <li><b>min-max scaling</b>, commonly called normalization
#             <br>
#             $$
#             \tilde{x} := \frac{x - x_{min}}{x_{max}-x_{min}} \in [0,1]
#             $$ which is in percentage units</li>
#         <br>
#             <li><b>standardization</b>
#             <br>
#             $$
#             \tilde{x} := \frac{x - \mathbb{E}[x]}{\sigma(x)}
#             $$ which is in standard deviation units</li>
#             <br>
#         </ul>
#         <li>Unlike max-min scaling, standardization does not bound values to a specific range and is much less affected by outliers because the range (denominator of max-min scaling) is greatly affected by outliers.</li>
#         <br>
#         <li style="color:red">As with all transformations, to prevent data leakage, it is important to fit the scalers to the training data only, and not to the full data set! Only then can you use them to transform the training set and test set.
#     </ul>
# </font>

# In[101]:


from sklearn.preprocessing import StandardScaler
#dataset is too small, so you need more data to make training data
#ie) if there was originally 100 entries, scaling would increase it to 1000

std_scale = StandardScaler()

std_scale.fit_transform(X_train.iloc[:, :-1])


# In[102]:


(X_train.iloc[:, :-1] - X_train.iloc[:, :-1].mean()) / X_train.iloc[:, :-1].std()


# In[103]:


from sklearn.preprocessing import MinMaxScaler

min_max_scale = MinMaxScaler()

min_max_scale.fit_transform(X_train.iloc[:, :-1])
#all the rows and all the columns except for the last 


# In[104]:


(X_train - X_train)/X_train.max()


# In[99]:


(X_train.iloc[:, :-1] - X_train.iloc[:, :-1].min()) / (X_train.iloc[:, :-1].max() - X_train.iloc[:, :-1].min())


# <h2>Some Reminders</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>Though we did the data preparation manually, it is good practice to write functions for several good reasons.</li>
#         <br>
#         <ul>
#         <li>It allows you to reproduce the transformations easily on any data set, for example the next time you get a fresh data set.</li>
#         <br>
#         <li>You will gradually build a library of transformation functions that you can reuse in future projects.</li>
#         <br>
#         <li>You can reuse these functions in production, i.e. on your live system, to transform the new data before feeding it to your models.</li>
#         <br>
#             <li>It will make it possible for you to easily try various transformations and see which combination of transformations works best.</li>
#             <br>
#             <li>It allows for better tinkering!</li>
#             <br>
#         </ul>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Putting It All Together</h2>

# In[63]:


X_train = train_set.drop('median_house_value', axis=1)  # drop already creates a copy

Y_train = train_set.loc[:,'median_house_value'].copy()


# In[64]:


X_train.info()


# In[65]:


X_train_numerical = X_train.iloc[:,:-1]
X_train_categorical = X_train.iloc[:, -1]


# In[66]:


# Handling missing values

median_train_set = X_train.loc[:,'total_bedrooms'].median()
X_train_numerical.loc[:,'total_bedrooms'].fillna(median_train_set, inplace=True)  


# In[67]:


# Creating the new variables

X_train_numerical['rooms_per_household'] = X_train_numerical['total_rooms'] / X_train_numerical['households']

X_train_numerical['bedrooms_per_room'] = X_train_numerical['total_bedrooms'] / X_train_numerical['total_rooms']

X_train_numerical['population_per_household'] = X_train_numerical['population'] / X_train_numerical['households']


# In[68]:


# Putting all the numerical variables on the same scale

X_train_numerical_scaled = (X_train_numerical - X_train_numerical.mean()) / X_train_numerical.std()


# In[69]:


# Handling categorical variable 

from sklearn.preprocessing import OneHotEncoder

dummy_encoder = OneHotEncoder()

ocean_proximity_dummy = dummy_encoder.fit_transform(X_train.loc[:, 'ocean_proximity'].values.reshape(-1,1))


# In[70]:


X_train_cleaned = pd.concat([X_train_numerical_scaled.reset_index(drop=True), 
                     pd.DataFrame(ocean_proximity_dummy.toarray(), columns=dummy_encoder.categories_[0])],
                    axis=1)


# In[71]:


X_train_cleaned


# In[72]:


train_set.drop('median_house_value', axis=1).iloc[:,:-1].mean()


# In[73]:


#this is basically everything we did above 
def clean_data(data_set, train_set):
    X = data_set.drop('median_house_value', axis=1)  # drop already creates a copy
    train_set = train_set.drop('median_house_value', axis=1)  # drop already creates a copy

    X_numerical = X.iloc[:,:-1]
    X_categorical = X.iloc[:, -1]
    train_set_numerical = train_set.iloc[:,:-1]
    
    # Handling missing values
    median_train_set = train_set.loc[:,'total_bedrooms'].median()
    X_numerical.loc[:,'total_bedrooms'].fillna(median_train_set, inplace=True)  
    #train_set_numerical.loc[:,'total_bedrooms'].fillna(median_train_set, inplace=True)  

    # Creating the new variables
    for df in [X_numerical, train_set_numerical]:
        df['rooms_per_household'] = df['total_rooms'] / df['households']
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
        df['population_per_household'] = df['population'] / df['households']

    # Putting all the numerical variables on the same scale
    X_numerical_scaled = (X_numerical - train_set_numerical.mean()) / train_set_numerical.std()

    # Handling categorical variable 
    from sklearn.preprocessing import OneHotEncoder
    dummy_encoder = OneHotEncoder()
    ocean_proximity_dummy = dummy_encoder.fit_transform(X_categorical.values.reshape(-1,1))

    X_cleaned = pd.concat([X_numerical_scaled.reset_index(drop=True), 
                         pd.DataFrame(ocean_proximity_dummy.toarray(), columns=dummy_encoder.categories_[0])],
                        axis=1)

    return X_cleaned


# In[74]:


clean_data(train_set, train_set)


# In[75]:


clean_data(test_set, train_set)


# <hr style="border: 20px solid black">

# <h1>Select and Train a Model</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>Finally, we have:</li>
#         <br>
#         <ul>
#             <li>framed the problem,</li>
#             <br>
#             <li>got the data,</li>
#             <br>
#             <li>explored the data,</li>
#             <br>
#             <li>sampled a training and test set,</li>
#             <br>
#             <li>explored transformations and created a pipeline to prepare the data to be fed into a predictive model.</li>
#             <br>
#         </ul>
#         <li>We're now ready to select and train an ML model.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Recall: In-Sample, Out-of-Sample</h2>
# <br>
# <font size="+1">
#     <br>
#     <ul>
#         <li>This is also known as making use of <b><i>holdout sets</i></b>.</li>
#         <br>
#         <li>We should hold back some of the data to check the model's performance.</li>
#         <br>
#         <li>We hope this holdout set is similar to unknown future data.</li>
#     </ul>
# </font>

# <h3>Model-Selection</h3>
# 
# <img src="data:image/png;base64,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" width="500" height="500" align="center"/>
# 

# <font size="+1">
#     <ul>
#         <li>We can do this splitting using a utility function from Scikit-Learn called <i><a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split</a></i>.</li>
#     <br>
# <li>Scikit-Learn's <i>train_test_split</i> utility function splits the data into random train and test subsets.</li>
#         <br>
#     </ul>
# </font>

# <font size="+0">
#     $$
#     \widehat{\varepsilon}^{IS} < \widehat{\varepsilon}^{OOS}.
#     $$
# </font>

# <h3>Hyperparameter-Selection</h3>
# 
# <img src="https://www.datavedas.com/wp-content/uploads/2018/04/image003.jpg" width="500" height="500" align="center"/>

# <font size="+1">
#     <ul>
#         <li>We have previously used a <i>holdout</i> set to estimate generalization out-of-sample error.</li>
#         <br>
#         <li>We can use another <i>holdout</i> set within the train (in-sample) set to test different hyperparameters.</li>
#         <br>
#         <li>This in-sample holdout set is known as a <i>validation</i> set and is used to estimate the generalization error of different hyperparameters.</li>
#         <br>
#         <li>Using a holdout set on the in-sample training set after you have used a holdout set on the full sample is known as <i>nested</i> validation.</li>
#         <br>
#     </ul>
# </font>

# <img src="https://i.stack.imgur.com/pXAfX.png" height="500" width="500" align="center"/>

# <font size="+1">
#     <ul>
#         <li>We will use the inner holdout set to find the hyperparameter corresponding to the best performing model.</li>
#         <br>
#         <li>Below, we use the outer holdout set to test the fitted model's generalization error. </li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Estimating and Evaluating In-Sample To Find A Model Class</h2>
# <br>
# <font size="+1">
#     Training and Evaluating on the Training Set:
#     <br>
#     <ul>
#         <li>Let's first train a Linear Regression model.</li>
#         <br>
#     </ul>
# </font>

# In[76]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train_cleaned, Y_train)


# In[ ]:





# <br>
# <font size="+1">
#     <ul>
#         <li>Done!</li>
#         <br>
#         <li>Now let's try it out on a few observations from the training set (in-sample).</li>
#         <br>
#     </ul>
# </font>

# In[77]:


some_in_sample_data = X_train_cleaned.iloc[:5, :]
some_in_sample_target_data = Y_train.iloc[:5]


# In[78]:


print("Predictions: {}".format(lin_reg.predict(some_in_sample_data)))

print("Targets: {}".format(list(some_in_sample_target_data)))


# <br>
# <font size="+1">
#     <ul>
#         <li>Although the predictions work, they aren't exactly accurate.</li>
#         <br>
#         <li>Let's measure this regression model's RMSE on the whole training set.</li>
#         <br>
#     </ul>
# </font>

# In[79]:


from sklearn.metrics import mean_squared_error

Y_hat_train = lin_reg.predict(X_train_cleaned)
np.sqrt(mean_squared_error(Y_train, Y_hat_train))


# <br>
# <font size="+1">
#     <ul>
#         <li>Most districts' median housing value is between 120k and 265k, so a typical prediction error of 68k isn't very satisfying.</li>
#         <br>
#         <li>This is an example of a model underfitting the training data.</li>
#         <br>
#         <li>When this happens it can mean that the variables do not provide enough information to make good predictions, or that the model isn't powerful enough.</li>
#         <br>
#         <li>The main ways to fix underfitting is to select a more powerful model, to feed the model better predictor variables, or to reduce the constraints on the model.</li>
#         <br>
#         <li>This model isn't regularized, which rules out the last option.</li>
#         <br>
#         <li>We could try to add more variables and do better feature engineering, but for now let's try a more complex model.</li>
#         <br>
#         <li>Specifically, we'll try a decision tree regressor that is capable of finding complex nonlinear relationships in the data.</li>
#         <br>
#     </ul>
# </font>

# In[80]:


#because the model above doesn't work for us, we decide to try to use a decision tree regressor
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(X_train_cleaned, Y_train)


# In[81]:


print("Predictions: {}".format(tree_reg.predict(some_in_sample_data)))

print("Targets: {}".format(list(some_in_sample_target_data)))


# In[82]:


from sklearn.metrics import mean_squared_error

Y_hat_train = tree_reg.predict(X_train_cleaned)
np.sqrt(mean_squared_error(Y_train, Y_hat_train))


# <br>
# <font size="+1">
#     <ul>
#         <li>Is the model perfect? Impossible! </li>
#         <br>
#         <li>This a clear example of a model being badly overfit to data.</li>
#         <br>
#         <li>How can we be sure since we haven't tested on our out-of-sample test data set?</li>
#         <br>
#         <li>We don't want to touch the test set until we're ready to launch a model we have confidence in.</li>
#         <br>
#         <li>This implies we need to use part of the training set for training and part of it for model validation.</li>
#         <br>
#         <li>That is, further split the training data set.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Better Evaluation Using Resampling Techniques To Find A Model Class</h2>

# <h3>Cross-Validation</h3>
# <font size="+1">
# <br>
#     <ul>
#         <li>When using a holdout set for estimating model generalization error and for choosing optimal hyperparameters, we lose a portion of our data which was held back for out-of-sample testing.</li>
#         <br>
#         <li>We also compute error over a fixed split, meaning we could be finding "optimal" settings which are only optimal to the specific split. This can be dangerous depending on the setting.</li>
#         <br>
#         <li>One way to address this problem is through a resampling technique known as <i>cross-validation</i>.</li>
#         <br>
#         <li>Cross-validation consists of a sequence of estimations where each subset of data is used as both an in-sample estimation set, and as an out-of-sample test set.</li>
#         <br>
#         <li>This sequence of estimations generates a sequence of error estimates. </li>
#         <br>
#         <li>We would like to combine this sequence of errors by some sort of weighted (possibly equal weighted) average to get a better measure of generalization error.</li>
#         <br>
#     </ul>
# <br>
# </font>
# <br>

# <h3>K-Fold Cross-Validation</h3>
# 
# 
# <img src="https://www.datarobot.com/wp-content/uploads/2018/03/Screen-Shot-2018-03-21-at-4.26.53-PM.png" width="500" height="500" align="center"/>

# <br>
# <font size="+1">
#     <ul>
#         <li>The idea is to split the data into $K$-many groups, and use each of group to to evaluate the model fit on the other $K-1$ groups.</li>
#         <br>
#         <li>Repeating this process for every group will yield a sequence of $K$-many errors, which can be averaged together to get a single, less noisy, estimate for error.</li>
#         <br>
#         <li>Just as the notion of <i>holdout</i> sets can be applied to the problem of model out-of-sample generalization error as well as hyperparameter error, so can the resampling technique of <i>cross-validation</i> be applied model error and hyperparameter error.</li>
#         <br>
#         <li>Rather than doing this manually, we can use Scikit-Learn's utility function <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html">cross_val_score</a>.</li>
#         <br>
#     </ul>
# </font>

# <img src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png" height="500" width="500" align="center"/>

# <font size="+1">
#     <ul>
#         <li>Model-Selection (Outer Fold)</li>
#         <br>
#         <li>Hyperparameter-Selection (Inner Fold)</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h3>Cross-Validation on In-Sample Data To Find A Model Class</h3>
# <br>
# <font size="+1">
#     <ul>
#         <li>One way to evaluate the Decision Tree model would be to use the <i>train_test_split()</i> method to split the training set into a smaller training and validation set. Then train your models against the smaller training set and evaluate them against the validation set.</li>
#         <br>
#         <li>A great alternative is Scikit-Learn's <i>K-fold cross-validation</i> functionality.</li>
#         <br>
#         <li>We will split the training set into 10 distinct subsets called <i>folds</i>.</li>
#         <br>
#         <li>Subsequently, the model gets estimated and evaluated 10 times, picking a different fold for evaluation every time while training on the other 9 folds. </li>
#         <br>
#         <li>The result is an array containing the 10 evaluation scores.</li>
#         <br>
#     </ul>
# </font>

# In[83]:


# Cross-validation can often be a bottleneck to you code
# Sklearn's implementation of cross validation scores expects a "utility/gain function" 
# rather than a "loss/cost function"

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, X_train_cleaned, Y_train, 
                                scoring="neg_mean_squared_error",
                                cv=10)

tree_rmse_in_sample_estimates = np.sqrt(-scores)

tree_rmse_in_sample_estimates


# In[84]:


# Let's look at the results

def display_scores(scores):
    print("==================================")
    print("Scores: {}".format(scores))
    print("==================================")
    print("Mean Score: {}".format(scores.mean()))
    print("==================================")
    print("Standard Deviation of Scores: {}".format(scores.std()))
    print("==================================")
    
    return None


# In[85]:


display_scores(tree_rmse_in_sample_estimates)


# <font size="+1">
#     <ul>
#         <li>Now the Decision Tree doesn't look as good as it did earlier.</li>
#         <br>
#         <li>In fact, it seems to perform worse than the Linear Regression model.</li>
#         <br>
#         <li>Notice that cross-validation allows you to not only get an estimate of the performance of your model, but also a measure of precise this estimate is, by its standard deviation.</li>
#         <br>
#         <li>You would not have this information if you just used a single validation set.</li>
#         <br>
#         <li>But cross-validation comes at the cost of training the model several times, which might not always be possible due to computational limits.</li>
#         <br>
#     </ul>
# </font>

# In[86]:


scores = cross_val_score(lin_reg, X_train_cleaned, Y_train, 
                                scoring="neg_mean_squared_error",
                                cv=10)

lin_reg_rmse_in_sample_estimates = np.sqrt(-scores)

display_scores(lin_reg_rmse_in_sample_estimates)


# In[87]:


# Score on only the in-sample training data set, not on the validation data sets

Y_hat_train = lin_reg.predict(X_train_cleaned)
np.sqrt(mean_squared_error(Y_train, Y_hat_train))


# <font size="+1">
#     <ul>
#         <li>Decision Tree model is overfitting so badly it is performing worse than the Linear Regression model.</li>
#         <br>
#         <li>Let's try one more model, the Random Forest Regressor.</li>
#         <br>
#         <li>A Random Forest model works by training many Decision Trees on random subsets of the predictor variables and the training data set, it then averages out their predictions.</li>
#         <br>
#         <li>Building a model on top of many other models is called <i>ensemble learning</i> and it is often a great way to push models further.</li>
#     </ul>
# </font>

# In[88]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(X_train_cleaned, Y_train)

scores = cross_val_score(forest_reg, X_train_cleaned, Y_train, 
                                scoring="neg_mean_squared_error",
                                cv=10)

forest_rmse_in_sample_estimates = np.sqrt(-scores)

forest_rmse_in_sample_estimates


# In[89]:


display_scores(forest_rmse_in_sample_estimates)


# In[90]:


# Score on only the in-sample training data set, not on the validation data sets

Y_hat_train = forest_reg.predict(X_train_cleaned)
np.sqrt(mean_squared_error(Y_train, Y_hat_train))


# <font size="+1">
#     <ul>
#         <li>This appears to be much better!</li>
#         <br>
#         <li>Notice that the score on the validation data set is still much lower than the score on in-sample training data set. This means the model is still overfitting the data set.</li>
#         <br>
#         <li style="color:red">TODO: Possible solutions for overfitting are to simplify the model, to constrain the model (i.e. regularize it), or get a lot more training data.</li>
#         <br>
#         <li style="color:red">TODO: At this stage, the goal is to try out many other models from various categories of ML algorithms, for example several Support Vector Machines with different kernels, without spending too much time tweaking the hyperparameters. We want to shortlist a few (two to five) promising model classes.</li>
#         <br>
#         <li>We will try out one more model - eXtreme Gradient BOOSTed trees - XGBoost.</li>
#         <br>
#     </ul>
# </font>

# In[91]:


import xgboost

xgb_reg = xgboost.XGBRegressor()

X_train_cleaned_xgb = X_train_cleaned.rename(columns={'<1H OCEAN': 'less_than_1H Ocean'})
xgb_reg.fit(X_train_cleaned_xgb, Y_train)

scores = cross_val_score(xgb_reg, X_train_cleaned_xgb, Y_train, 
                                scoring="neg_mean_squared_error",
                                cv=10)

xgb_rmse_in_sample_estimates = np.sqrt(-scores)

xgb_rmse_in_sample_estimates


# In[ ]:


display_scores(xgb_rmse_in_sample_estimates)


# In[ ]:


# Score on only the in-sample training data set, not on the validation data sets

Y_hat_train = xgb_reg.predict(X_train_cleaned_xgb)
np.sqrt(mean_squared_error(Y_train, Y_hat_train))


# <font size="+1">
#     <ul>
#         <li>This appears to be even better than the Random Forest regressor, and overfits less as well!</li>
#         <br>
#         <li>As a rule of thumb, Gradient Boosted Trees tend to be "the best" when it comes to tabular structured data sets.</li>
#         <br>
#     </ul>
# </font>

# <font size="+1" style="color:orange">
#     <ul>
#         <b>Not Necessary To Know</b>
#         <br>
#         <li>In practice, you should save every model you experiment with so that you can come back easily to any model you want without having to re-estimate your models.</li>
#         <br>
#         <li>Depending on your application you may want to save different things, but generally you will want to save:</li>
#         <br>
#         <ul>
#             <li>hyperparameters</li>
#             <br>
#             <li>estimated parameters</li>
#             <br>
#             <li>cross-validation scores</li>
#             <br>
#             <li>predicted values</li>
#             <br>
#         </ul>
#         <li>This will easily allow you to compare scores across model types and compare the types of errors they make.</li>
#     </ul>
# </font>

# In[ ]:


#plt.plot(X_train_cleaned['median_income'],Y_train, '.')
#plt.plot(X_train_cleaned['median_income'], Y_hat_train, '.')

#plt.plot(Y_train, Y_hat_train, '.')


# <hr style="border: 20px solid black">

# <h1>Fine-Tuning Your Model</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>At this point, let's assume we have a shortlist of promising models. Now we need to fine-tune them.</li>
#         <br>
#     </ul>
# </font>

# <h2>Grid Search</h2>
# <br>
# <br>
# <img src="https://i.ytimg.com/vi/pKxrm26ACa4/maxresdefault.jpg">
# <br>
# <br>
# <br>
# <br>
# <font size="+1">
#     <ul>
#         <li>One way to choose the best hyperparameter settings is to manually change the settings until you find a sufficient combination of hyperparameter values.</li>
#         <br>
#         <li>This is tedious and time consuming!</li>
#         <br>
#         <li>Instead, you can use Scikit-Learn's built in <i>GridSearchCV</i> to do the searching and selection for you.</li>
#         <br>
#         <li>All you need to do is specify the hyperparameters you wish to experiment with, and the values you want to investigate.</li>
#         <br>
#         <li>For example, we'll consider the best hyperparameter combination for the Random Forest Regressor.</li>
#         <br>
#     </ul>
# </font>

# In[92]:


# To see all the possible hyperparameters, see RandomForestRegressor?

from sklearn.model_selection import GridSearchCV

# The parameter grid tells the estimator to first evaluate all 3x4=12 combinations of 'n_estimators' and 'max_features'
# Then it tells the estimator to evaluate all 1x2x3=6 combinations of 'boostrap', 'n_estimators', and 'max_features'
param_grid = [
    {'n_estimators': [3, 10, 30],
     'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False],
     'n_estimators': [3, 10],
     'max_features': [2, 3, 4]}
]


# <font size="+1">
#     <ul>
#         <li>Don't worry about what all these hyperparameters mean for now, we will learn about them later.</li>
#     </ul>
# </font>

# In[93]:


forest_reg = RandomForestRegressor()

# The grid search will explore 12+6=18 combinations of RandomForestRegressor hyperparameter value,
# and it will train each model 5 times (since we're using 5-fold CV)
# resulting in 18x5=90 rounds of estimation!
grid_search = GridSearchCV(forest_reg,
                          param_grid,
                          cv=5,
                          scoring='neg_mean_squared_error',
                          return_train_score=True, 
                          refit=True)
# 'refit=True' means once grid search finds the best estimator using CV, 
# it re-estimates the model on the whole training set

grid_search.fit(X_train_cleaned, Y_train)


# <font size="+1">
#     <ul>
#         <li>When you have no prior belief about what a hyperparameter value should be, a simple approach is to try out consecutive powers of 10, or a smaller number if you want a more fine-grained search.</li>
#     </ul>
# </font>

# In[94]:


# When the grid search is finished, you can get the best combination of hyperparameters 
# Should we stop, or should we continue?

grid_search.best_params_


# In[95]:


# We can also get the best estimator directly

grid_search.best_estimator_


# In[96]:


# We can also view the evaluation scores

cv_results = grid_search.cv_results_

cv_results


# In[97]:


# Technique to create a double-iterable object from two iterable objects
zip(cv_results['mean_test_score'], cv_results['params'])


# In[98]:


for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(np.sqrt(-mean_score), params)


# <font size="+1">
#     <ul>
#         <li>How does the tuned-hyperparameter model's score compare to the default hyperparameter settings?</li>
#         <br>
#         <li style="color:red">Many data preparation steps can be treated as hyperparameters.</li>
#         <br>
#         <li style="color:red">As an example, you can use grid search to automatically find out whether or not to add a feature you were not sure about using. It can similarly be used to find the best way to handle outliers, missing variables, variable selection, and more.</li>
#         <br>
#     </ul>
# </font>

# <h2>Randomized Search</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li><img src="Images/grid_search_vs_random_search.png"></li>
#         <br>
#         <li>The grid search approach is fine when you are exploring relatively few combinations, but when the hyperparameter search space is large, it is often preferable to use <i>RandomizedSearchCV</i> instead.</li>
#         <br>
#         <li>It works by evaluating a given number of random combinations by selecting a random value for each hyperparameter at every iteration.</li>
#         <br>
#         <li style="color:red">TODO: Explore this option.</li>
#         <br>
#     </ul>
# </font>

# <h2>Ensemble Methods</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>Another way to fine-tune your system is to try to combine the models that perform best.</li>
#         <br>
#         <li>The group (or "ensemble") will often perform better than the best individual model, especially if the individual models make very different types of errors.</li>
#         <br>
#         <li style="color:red">TODO: Explore this option.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h2>Analyze the Best Models and Their Errors</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>You will often gain good insights on the problem by inspecting the best models.</li>
#         <br>
#         <li>For example, the <i>RandomForestRegressor</i> can indicate the relative importance of each attribute for making accurate predictions:</li>
#         <br>
#     </ul>
# </font>

# In[ ]:


feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances


# In[ ]:


# We can display these importance scores next to their corresponding attribute names

plt.figure(figsize=(16,8))
plt.barh(X_train_cleaned.columns, feature_importances)


# In[ ]:


for predictor, importance in zip(X_train_cleaned.columns, feature_importances):
    print(predictor, "              ", importance)


# In[ ]:


sorted(zip(feature_importances, X_train_cleaned.columns), reverse=True)


# <br>
# <font size="+1">
#     <ul>
#         <li>With this information, you may want to consider dropping some of the less useful features, for example only one <i>ocean_proximity</i> category is really useful.</li>
#         <br>
#     </ul>
# </font>

# In[ ]:


# Add a section about Shapley values


# <hr style="border: 20px solid black">

# <h2>Evaluate Your Model Out-of-Sample</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>After tweaking your models for a while, you eventually have a system that performs sufficiently well.</li>
#         <br>
#         <li>Now it is time to evaluate the final model on the out-of-sample test set.</li>
#         <br>
#     </ul>
# </font>

# In[ ]:


test_set


# In[ ]:


X_test_cleaned = clean_data(test_set, train_set)
Y_test = test_set['median_house_value'].copy()


# In[ ]:


final_model = grid_search.best_estimator_

Y_hat_test = final_model.predict(X_test_cleaned)

final_mse = mean_squared_error(Y_test, Y_hat_test)

final_rmse = np.sqrt(final_mse)


# In[ ]:


final_rmse


# <br>
# <font size="+1">
#     <ul>
#         <li>In some cases, such a point estimate of the generalization error will not be enough to convince you to launch the model in production.</li>
#         <br>
#         <li>What if it is just $0.1\%$ better than the model in production? What does this score improvement correspond to in business revenue?</li>
#         <br>
#         <li>Furthermore, you might want to have an idea of how precise this estimate is. For this, you can compute a <i><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html">confidence interval</a></i> for the generalization error.</li>
#         <br>
#         <li style="color:red">TODO: As an exercise, in words, explain what a confidence interval for the point estimate of the generalization error is capturing to a product manager or a potential client.</li>
#         <br>
#         <li style="color:red">TODO: It might be a good idea to consider <i>uncertainty</i> models and measurements, as we've just considered models optimized for <i>accuracy</i>.</li>
#         <br>
#     </ul>
# </font>

# <img src="https://i.stack.imgur.com/FdKec.jpg" width="600" height="600">

# <img src="https://i.pinimg.com/originals/47/d5/67/47d567be818bf7ca775a9370acb1517b.jpg">

# In[ ]:


# To calculate a 95% confidence interval for the point estimate of the generalization out-of-sample error, 
# we'll use Scipy and more-or-less do the computation manually.

from scipy import stats

confidence = 0.95
squared_errors_oos = (Y_test - Y_hat_test)**2

confidence_interval = np.sqrt(stats.t.interval(alpha=confidence,                     # confidence level
                         df=len(squared_errors_oos) -1,        # degrees of freedom
                         loc=squared_errors_oos.mean(),        # location of the t-distribution
                         scale=stats.sem(squared_errors_oos))) # scale of the t-distribution set to be the 
                                                               # std error of the mean of the out-of-sample squared errors


# In[ ]:


print(f'The point estimate for the average test set (OOS) error is {np.round(np.sqrt(squared_errors_oos.mean()),0)} and the confidence interval is {np.round(confidence_interval,0)}.')


# In[ ]:


# stats.t.interval?
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html # std error of mean of a sample


# <br>
# <font size="+1">
#     <ul>
#         <li>If you did a lot of hyperparameter tuning, the performance will usually be slightly worse than what you measured using cross-validation.</li>
#         <br>
#         <li>This is due to your model being estimated to perform well on the validation data set, and will likely not perform as well on unknown data sets.</li>
#         <br>
#         <li>This isn't the case in this problem, but when it happens its important to not tweak the hyperparameters to make the errors look good on the test set as the improvement would be unlikely to generalize well to new data in production.</li>
#         <br>
#         <li style="color:red">TODO: The project now enters <i>pre-launch</i> phase where you need to present your solution</li>
#         <br>
#         <ul style="color:red">
#             <li>highlighting what you have learned, </li>
#             <br>
#             <li>describing what experiments you conducted, and which you did not conduct,</li>
#             <br>
#             <li>what assumptions were made,</li>
#             <br>
#             <li>and what your model's limitations are.</li>
#             <br>
#         </ul>
#         <li style="color:red">TODO: You also need to document everything, and create nice presentations with clear visualizations and easy-to-remember statements, such as <i>"the median income is the number one predictor of housing prices"</i>. That is you need to highlight your OKRs (objectives and key results), that is goals and deliverables.</li>
#         <br>
#         <li style="color:red">TODO: With the help of a project or product manager, you need to translate your results into business or product results, including translating score improvements into dollar values.</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# <h1>In Summary</h1>
# <br>
# <font size="+1">
#     <ol>
#         <li>Define our business objective and decide how to measure success.</li>
#         <br>
#         <li>Split the data to prevent overfitting.</li>
#         <br>
#         <li>Explore the data (EDA) to gain insights that can help achieve our business objective. This includes coming up with new predictive variables.</li>
#         <br>
#         <li>Prepare the data for building a predictive model.</li>
#         <br>
#         <li>Select and estimate (train) a model.</li>
#         <br>
#         <li>Fine tune your model!</li>
#         <br>
#     </ol>
# </font>

# <img src="https://i.ytimg.com/vi/1wVgtINZIT4/maxresdefault.jpg">

# <hr style="border: 20px solid black">

# <h1 style="color:blue">Describe the Problem and Solution As If You Were Talking to a Hiring Manager - Homework Exercise</h1>

# <hr style="border: 20px solid black">

# #
# <h1>Miscellaneous: NOT NECESSARY TO KNOW</h1>

# <h2><a href="https://www.sciencedirect.com/science/article/pii/S016771529600140X">Another Experiment</a></h2>
# <br>
# <br>
# 
# ![sparse_spatial_regressions.PNG](attachment:sparse_spatial_regressions.PNG)
# 
# <br>
# <font size="+1">
#     <ul>
#         <li></li>
#         <br>
#     </ul>
# </font>

# <h2>Launch Your System</h2>
# <br>
# <font size="+1" style="color:blue">
#     <ul>
#         <li>Assuming you <i>pre-launch phase</i> went successful and the business stakeholders gave you approval to launch your model, you now need to get your model ready for production.</li>
#         <br>
#         <li>This can include: polishing/re-factoring your code into functions, classes, and modules, writing adequate documentation, writing tests, etc.; eventually deploying your model to your production environment.</li>
#         <br>
#         <li>One way to do this is to save your model using <i>joblib</i> or some other <i>JSON</i> file formatting, which includes the full preprocessing and prediction pipeline, then load this trained model within your production environmet and use it to make predictions by calling <i>predict()</i></li>
#         <br>
#         <li>To avoid any environment issues, you should containerize your saved model using a Docker Image.</li>
#         <br>
#         <li> For example, perhaps the model will be used within a website where the user will type in some data about a new district and click an <i>Estimate Price</i> button.</li>
#         <br>
#         <li>This will send a query containing the data to the web server, which will forward the data to your web application, and finally your code will simply call the model's <i>predict()</i> method. You want to load the model upon server startup, rather than every time the model is used.</li>
#         <br>
#         <li>Alternatively, you can wrap the model within a dedicated web service that your web application can query through a REST API. This makes it easier to upgrade your model to new versions without interrupting the main application. It also simplifies scaling, since you can start as many web services as needed and load-balance the requests coming from your web application across these web services. It also allows your web application to use any language, not just Python.</li>
#         <br>
#         <li>Another popular strategy is to deploy your estimated model to a remote (cloud) server, for example an AWS EC2-instance, or Google Cloud AI platform. You can just save your model and upload it to remote (cloud) storage, then head over to the remote server and create a new model version pointing to the file on the remote storage. This application takes JSON requests containint the input data of a district and returns JSON responses containing the predictions.</li>
#         <br>
#     </ul>
# </font>

# In[ ]:





# <h2>Monitor Your System</h2>
# <br>
# <font size="+1" style="color:blue">
#     <ul>
#         <li>Deployment is not the end of the story!</li>
#         <br>
#         <li>You also need to write monitoring code to check your system's live performance at regular intervals and trigger alerts when it drops.</li>
#         <br>
#         <li>This could be a rapid and steep drop, likely due to a broken component in your infrastructure, but be aware as it could be a slow and steady decline that could easily go unnoticed for a long time.</li>
#         <br>
#         <li>This is quite common because models tend to <i>rot</i> over time due to data changing. If the model was estimated on last year's data, it may not be suited to predict today's data.</li>
#         <br>
#         <li>This means you need to monitor your model's live performance. This is another art form.</li>
#         <br>
#         <li>In some cases, the model's performance can be inferred from downstream metrics. For example, if the model is part of a recommender system, and it suggests products that the users may be interestred in, then it is easy to monitor the number of recommended products sold each day. If this number drops, compared to the recommended products, then the prime suspect is the model.</li>
#         <br>
#         <li>You need to put in place a monitoring system as well as the relevant processes to define what to do in case of failures and how to prepare for them. <b>THIS CAN BE A LOT OF WORK, OFTEN MORE THAN THE BUILDING AND ESTIMATION OF A MODEL.</b></li>
#         <br>
#         <li>If the data keeps evolving (very common in financial data), you will need to pudate your datasets and retrain your model regularly. You should probably automate the whole process as best you can.</li>
#         <br>
#         <li>You also need to make sure you evaluate the model's input data quality. For example, you could trigger alerts if more and more observations are missing a variable, or if a variable's mean or standard deviation drifts too far away from the training set, or a categorical variable starts containing new categories.</li>
#         <br>
#     </ul>
# </font>

# In[ ]:





# <h2>Maintain Your System</h2>
# <br>
# <font size="+1" style="color:blue">
#     <ul>
#         <li>Finally, make sure to keep backups of every model you create and have the process and tools in place to roll back to a previous model quickly, in the event of a new model failing badly.</li>
#         <br>
#         <li>Having backups also makes it easy to compare new models with previous ones.</li>
#         <br>
#         <li>Similarly you should backup all versions of your datasets in case any one needs to be rolled back to a previous version due to some data generation error.</li>
#         <br>
#         <li>You may want to create several subsets of the test set in order to evaluate how well your model performs on specific parts of the data. For example, you may want to have a subset containing only the most recent data, or a test set for specific kinds of observations (i.e. districts located inland versus districts located near the ocean).</li>
#         <br>
#         <li>This will give you a deeper understanding of your model's strengths and weaknesses.</li>
#         <br>
#         <li>Clearly, machine learning involves quite a lot of <b>infrastructure</b>. </li>
#         <br>
#     </ul>
# </font>

# In[ ]:





# In[ ]:




