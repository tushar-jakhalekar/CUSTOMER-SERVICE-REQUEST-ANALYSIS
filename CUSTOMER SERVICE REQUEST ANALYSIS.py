#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER SERVICE REQUEST ANALYSIS

# #### Import the required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# #### Load the csv file

# In[2]:


df = pd.read_csv('T:\Masters In Data Science\Data Science with Python\Project\Customer_Service_Requests.csv')


# In[3]:


df.head() ## Checking first 5 rows of the data


# In[4]:


df.shape  ## Checking the shape


# In[5]:


df.size  ## Checking the size


# In[6]:


df.describe  ## Checking the summary of the data


# In[7]:


df1 = df.fillna(0)  ## Filling the NAN values with 0


# In[8]:


df1.head()


# #### Convert the columns ‘Created Date’ and Closed Date’ to datetime datatype

# In[9]:


df1["Created_Date"] = pd.to_datetime(df1["Created Date"])
df1["Closed_Date"] = pd.to_datetime(df1["Closed Date"])


# #### Create a new column ‘Request_Closing_Time’ as the time elapsed between request creation and request closing.

# In[10]:


df1['Request_Closing_Time'] = df1['Closed_Date'] - df1['Created_Date']


# In[11]:


df1.head(10)


# #### Provide major insights/patterns that you can offer in a visual format (graphs or tables)

# In[12]:


df1 = df1.set_index(df1['Created_Date'])
df1.groupby(pd.Grouper(freq = 'M')).size().plot()


# In[13]:


df1_type = df1['Complaint Type'].str.upper().value_counts() 
## Getting value counts of Registered complaints by Complaint Type
df1_type


# In[14]:


df1_location_type = df1['Location Type'].str.upper().value_counts()   
## Getting value counts of Registered complaints by LocationType
df1_location_type


# In[15]:


df1_type.head().plot(kind = 'pie', autopct = '%0.2f%%', figsize = (6,5))
plt.axis('equal')
plt.title('Complaint Type Distribution')
plt.tight_layout()
plt.show()


# #### Pie chart shows type of complaint which is registered most

# In[16]:


df1_location_type.head().plot(kind = 'pie', autopct = '%0.2f%%', figsize = (9,8))
plt.axis('equal')
plt.title('Location Type Distribution')
plt.tight_layout()
plt.show()


# #### Pie chart shows Location type of complaint which is registered most

# In[17]:


df1_descriptor = df1['Descriptor'].str.upper().value_counts()  
## Getting value counts of Registered complaints by Complaint Type
df1_descriptor


# In[18]:


df1_descriptor.head().plot(kind = 'pie', autopct = '%0.2f%%', figsize = (6,5))
plt.axis('equal')
plt.title('Descriptor Distribution')
plt.tight_layout()
plt.show()


# #### Pie chart shows type of complaint which is registered most

# In[19]:


df1_zip = df1['Incident Zip'].value_counts()
## Getting value counts of Registered complaints by Zip code
df1_zip


# In[20]:


df1_zip.head().plot(kind = 'pie', autopct = '%0.2f%%', figsize = (6,5))
plt.axis('equal')
plt.title('Zip code wise distribution')
plt.tight_layout()
plt.show()


# #### Pie chart shows which zip codes have most complaints

# #### Order the complaint types based on the average ‘Request_Closing_Time’, grouping them for different locations.

# In[21]:


df1_avg_closing_time_zip = df1.groupby('Incident Zip').Request_Closing_Time.mean() 
## Getting the average closing time of complaints registered at different zip codes
df1_avg_closing_time_zip.head(50)


# In[22]:


df1_avg_closing_time_Location = df1.groupby('Location').Request_Closing_Time.mean()
## Getting the average closing time of complaints registered at different Locations
df1_avg_closing_time_Location.head(50)


# In[23]:


df1_avg_closing_time_Type = df1.groupby('Complaint Type').Request_Closing_Time.mean()
## Getting the average closing time of complaints registered at different Complaint Types
df1_avg_closing_time_Type.head(25)


# #### Whether the average response time across complaint types is similar or not (overall)

# In[24]:


df1_avg_closing_time_Type = pd.to_numeric(df1_avg_closing_time_Type)
## converting to numeric to get perfect mean value
df1_avg_closing_time_Type.mean()


# In[25]:


df2 = pd.to_numeric(df1['Request_Closing_Time'])
df2.shape


# #### Null hypothesis is "the average response time across complaint types is similar"
# 
# #### Alternate Hypothesis is " the average response time across complaint types is not similar"

# In[26]:


from scipy import stats
from statsmodels.stats import weightstats as stests


# In[27]:


ztest ,pval = stests.ztest(df2.head(1000), x2=None, value=df1_avg_closing_time_Type.mean())
print(float(pval))


# #### Here the p-value is less than Significance level (0.05) hence we reject the null hypothesis 
# 
# ### * The average response time across complaint types is not similar *

# #### Are the type of complaint or service requested and location related?

# #### H0,There is a relationship between Complaint Type and Location
# 
# #### H1,There is no relationship between Complaint Type and Location

# In[28]:


contingency_table=pd.crosstab(df1['Complaint Type'],df1['Location'])
print('contingency_table :-\n',contingency_table)


# In[29]:


#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)


# In[30]:


no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)
alpha = 0.05


# In[31]:


from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)


# In[32]:


#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between Complaint Type and Location")
else:
    print("Retain H0,There is no relationship between Complaint Type and Location")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between Complaint Type and Location")
else:
    print("Retain H0,There is no relationship between Complaint Type and Location")


# ### * There is a relationship between Complaint Type and Location *

# In[ ]:




