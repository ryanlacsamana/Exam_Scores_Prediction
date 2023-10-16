## **Exam Scores Prediction**

The dataset contains test scores from three different subjects at a fictional public school. This dataset also contains variables about various personal and socio-economic factors for each student.

The goal is to provide an analysis on how several variables could affect a student's exam score and to predict future exam scores given those variables.

*Credits to the creator of the original dataset, **Mr. Royce Kimmons**, and to the uploaded dataset in **[Kaggle](https://www.kaggle.com/datasets/desalegngeb/students-exam-scores/data)***.

### **Data Description**

Column | Description |
-----|-----|
Gender | Gender of the student (male/female) |
EthnicGroup | Ethnic group of the student (group A to E) |
ParentEduc | Parent(s) education background (from some_highschool to master's degree) |
LunchType | School lunch type (standard or free/reduced) |
TestPrep | Test preparation course followed (completed or none) |
ParentMaritalStatus | Parent(s) marital status (married/single/widowed/divorced) |
PracticeSport | How often the student practice sport (never/sometimes/regularly) |
IsFirstChild | If the first child in the family or not (yes/no) |
NrSiblings | Number of siblings the student has (0 to 7) |
TransportMeans | Means of transport to school (schoolbus/private) |
WklyStudyHours | Weekly self-study hours(less than 5hrs, between 5 and 10 hrs, more than 10 hours) |
MathScore | Math test score (0-100) |
ReadingScore | Reading test score (0-100) |
WritingScore | Writing test score (0-100) |

### **Preparation**

```
## For data manipulation

import numpy as np
import pandas as pd

## For data visualization

import matplotlib.pyplot as plt
import seaborn as sns

## For displaying all columns in the dataframe

pd.set_option('display.max_columns', None)

## For statistical tests

from scipy import stats

## For data modelling

import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

## For metrics and helpful functions

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RepeatedKFold

## Miscellaneous

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
```

### **DATA CLEANING**

```
## Load the dataset

df = pd.read_csv("/kaggle/input/students-exam-scores/Expanded_data_with_more_features.csv")

df.head(10)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Gender</th>
      <th>EthnicGroup</th>
      <th>ParentEduc</th>
      <th>LunchType</th>
      <th>TestPrep</th>
      <th>ParentMaritalStatus</th>
      <th>PracticeSport</th>
      <th>IsFirstChild</th>
      <th>NrSiblings</th>
      <th>TransportMeans</th>
      <th>WklyStudyHours</th>
      <th>MathScore</th>
      <th>ReadingScore</th>
      <th>WritingScore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>female</td>
      <td>NaN</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>regularly</td>
      <td>yes</td>
      <td>3.0</td>
      <td>school_bus</td>
      <td>&lt; 5</td>
      <td>71</td>
      <td>71</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>NaN</td>
      <td>married</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>5 - 10</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>single</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>4.0</td>
      <td>school_bus</td>
      <td>&lt; 5</td>
      <td>87</td>
      <td>93</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>married</td>
      <td>never</td>
      <td>no</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5 - 10</td>
      <td>45</td>
      <td>56</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>0.0</td>
      <td>school_bus</td>
      <td>5 - 10</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>female</td>
      <td>group B</td>
      <td>associate's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>regularly</td>
      <td>yes</td>
      <td>1.0</td>
      <td>school_bus</td>
      <td>5 - 10</td>
      <td>73</td>
      <td>84</td>
      <td>79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>female</td>
      <td>group B</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>widowed</td>
      <td>never</td>
      <td>no</td>
      <td>1.0</td>
      <td>private</td>
      <td>5 - 10</td>
      <td>85</td>
      <td>93</td>
      <td>89</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>male</td>
      <td>group B</td>
      <td>some college</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>married</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>1.0</td>
      <td>private</td>
      <td>&gt; 10</td>
      <td>41</td>
      <td>43</td>
      <td>39</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>male</td>
      <td>group D</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>completed</td>
      <td>single</td>
      <td>sometimes</td>
      <td>no</td>
      <td>3.0</td>
      <td>private</td>
      <td>&gt; 10</td>
      <td>65</td>
      <td>64</td>
      <td>68</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>female</td>
      <td>group B</td>
      <td>high school</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>married</td>
      <td>regularly</td>
      <td>yes</td>
      <td>NaN</td>
      <td>private</td>
      <td>&lt; 5</td>
      <td>37</td>
      <td>59</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>

```
## Check the size of the dataset

print(df.shape)
```
(30641, 15)
The dataset contains 30,641 rows and 15 columns.
```
## Check information about the dataset

df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30641 entries, 0 to 30640
Data columns (total 15 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   Unnamed: 0           30641 non-null  int64  
 1   Gender               30641 non-null  object 
 2   EthnicGroup          28801 non-null  object 
 3   ParentEduc           28796 non-null  object 
 4   LunchType            30641 non-null  object 
 5   TestPrep             28811 non-null  object 
 6   ParentMaritalStatus  29451 non-null  object 
 7   PracticeSport        30010 non-null  object 
 8   IsFirstChild         29737 non-null  object 
 9   NrSiblings           29069 non-null  float64
 10  TransportMeans       27507 non-null  object 
 11  WklyStudyHours       29686 non-null  object 
 12  MathScore            30641 non-null  int64  
 13  ReadingScore         30641 non-null  int64  
 14  WritingScore         30641 non-null  int64  
dtypes: float64(1), int64(4), object(10)
memory usage: 3.5+ MB
```
#### **Check for null values**
```
## Check the number of null values for each variable in the dataset

for col in df.columns:
    print('Null Values for column {} is {}%'.format(col, np.round(df[col].isnull().sum()*100 / len(df[col])),2))
```
Null Values for column Unnamed: 0 is 0.0%
Null Values for column Gender is 0.0%
Null Values for column EthnicGroup is 6.0%
Null Values for column ParentEduc is 6.0%
Null Values for column LunchType is 0.0%
Null Values for column TestPrep is 6.0%
Null Values for column ParentMaritalStatus is 4.0%
Null Values for column PracticeSport is 2.0%
Null Values for column IsFirstChild is 3.0%
Null Values for column NrSiblings is 5.0%
Null Values for column TransportMeans is 10.0%
Null Values for column WklyStudyHours is 3.0%
Null Values for column MathScore is 0.0%
Null Values for column ReadingScore is 0.0%
Null Values for column WritingScore is 0.0%

Some columns have null values. However, we will check the best method to handle those missing values.

Also, the Dtype for the column NrSiblings is float64. This is not the correct datatype because the column contains discrete variables. The datatype for this column should be changed to **int64**.

#### **Converting datatypes, renaming columns, and removing unnecessary columns**
```
## Changing the datatype of 'NrSiblings' into int64

df['NrSiblings'] = pd.to_numeric(df['NrSiblings'], downcast='integer', errors='coerce')
df['NrSiblings'] = df['NrSiblings'].astype('Int64')

df['NrSiblings'].info()
```

```
<class 'pandas.core.series.Series'>
RangeIndex: 30641 entries, 0 to 30640
Series name: NrSiblings
Non-Null Count  Dtype
--------------  -----
29069 non-null  Int64
dtypes: Int64(1)
memory usage: 269.4 KB
```
```
## Removed the 'Unnamed' column

df = df.drop('Unnamed: 0', axis=1)
```
```
## Change column names to 'snake_case'

df.rename(columns={'Gender':'gender',
                   'EthnicGroup':'ethnic_group',
                   'ParentEduc':'parents_education',
                   'LunchType':'lunch_type',
                   'TestPrep':'test_preparation',
                   'ParentMaritalStatus':'parent_marital_status',
                   'PracticeSport':'practice_sports',
                   'IsFirstChild':'is_first_child',
                   'NrSiblings':'number_of_siblings',
                   'TransportMeans':'transport_means',
                   'WklyStudyHours':'weekly_study_hours',
                   'MathScore':'math_score',
                   'ReadingScore':'reading_score',
                   'WritingScore':'writing_score'}, inplace=True)

df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ethnic_group</th>
      <th>parents_education</th>
      <th>lunch_type</th>
      <th>test_preparation</th>
      <th>parent_marital_status</th>
      <th>practice_sports</th>
      <th>is_first_child</th>
      <th>number_of_siblings</th>
      <th>transport_means</th>
      <th>weekly_study_hours</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>NaN</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>regularly</td>
      <td>yes</td>
      <td>3</td>
      <td>school_bus</td>
      <td>&lt; 5</td>
      <td>71</td>
      <td>71</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>NaN</td>
      <td>married</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>0</td>
      <td>NaN</td>
      <td>5 - 10</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>single</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>4</td>
      <td>school_bus</td>
      <td>&lt; 5</td>
      <td>87</td>
      <td>93</td>
      <td>91</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>married</td>
      <td>never</td>
      <td>no</td>
      <td>1</td>
      <td>NaN</td>
      <td>5 - 10</td>
      <td>45</td>
      <td>56</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>sometimes</td>
      <td>yes</td>
      <td>0</td>
      <td>school_bus</td>
      <td>5 - 10</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>

### **Handling Missing Data**
The method for handling missing data will be either by removing the null values, or by imputation.

#### **Method 1A: Removing null values in the dataset**
```
## Drop null values from all columns

df1 = df.dropna(axis=0)

df1.info()
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 19243 entries, 2 to 30640
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype 
---  ------                 --------------  ----- 
 0   gender                 19243 non-null  object
 1   ethnic_group           19243 non-null  object
 2   parents_education      19243 non-null  object
 3   lunch_type             19243 non-null  object
 4   test_preparation       19243 non-null  object
 5   parent_marital_status  19243 non-null  object
 6   practice_sports        19243 non-null  object
 7   is_first_child         19243 non-null  object
 8   number_of_siblings     19243 non-null  Int64 
 9   transport_means        19243 non-null  object
 10  weekly_study_hours     19243 non-null  object
 11  math_score             19243 non-null  int64 
 12  reading_score          19243 non-null  int64 
 13  writing_score          19243 non-null  int64 
dtypes: Int64(1), int64(3), object(10)
memory usage: 2.2+ MB
```
#### **Method 1B: Removing null values from specified columns**

To avoid over-representation of certain data, the limit for the number of null values in which imputation will be used will be set to 5.0% of the total number of data in a certain column.

The columns EthnicGroup, ParentEduc, TestPrep, and TransportMeans contains null values that are more than 5.0% of the total number of data in their respective column. We will remove null values from this columns before proceeding with imputation.
```
## Drop null values from selected columns

df1b = df.copy()

df1b = df1b.dropna(subset=['ethnic_group','parents_education','test_preparation','number_of_siblings','transport_means'], axis=0)

df1b.info()
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 21721 entries, 2 to 30640
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype 
---  ------                 --------------  ----- 
 0   gender                 21721 non-null  object
 1   ethnic_group           21721 non-null  object
 2   parents_education      21721 non-null  object
 3   lunch_type             21721 non-null  object
 4   test_preparation       21721 non-null  object
 5   parent_marital_status  20883 non-null  object
 6   practice_sports        21298 non-null  object
 7   is_first_child         21094 non-null  object
 8   number_of_siblings     21721 non-null  Int64 
 9   transport_means        21721 non-null  object
 10  weekly_study_hours     21038 non-null  object
 11  math_score             21721 non-null  int64 
 12  reading_score          21721 non-null  int64 
 13  writing_score          21721 non-null  int64 
dtypes: Int64(1), int64(3), object(10)
memory usage: 2.5+ MB
```
#### **Method 2: Use Mode Imputation**
The values for the columns with null data are contains categorical and binary data. To avoid over-representing certain data, we set a limit to null values equal to or less than 5.0% of the total number of data in their respective column. The columns with null values that exceeded the limit had all their null values dropped.
```
## Mode imputation for categorical columns

df1b['parent_marital_status'] = df1b['parent_marital_status'].fillna(df1b['parent_marital_status'].mode()[0])
df1b['practice_sports'] = df1b['practice_sports'].fillna(df1b['practice_sports'].mode()[0])
df1b['weekly_study_hours'] = df1b['weekly_study_hours'].fillna(df1b['weekly_study_hours'].mode()[0])

## Mode imputation for binary columns

df1b['is_first_child'] = df1b['is_first_child'].fillna(df1b['is_first_child'].mode()[0])

df1b.isnull().sum()
```
```
gender                   0
ethnic_group             0
parents_education        0
lunch_type               0
test_preparation         0
parent_marital_status    0
practice_sports          0
is_first_child           0
number_of_siblings       0
transport_means          0
weekly_study_hours       0
math_score               0
reading_score            0
writing_score            0
dtype: int64
```
```
df1b.info()
```
```
<class 'pandas.core.frame.DataFrame'>
Index: 21721 entries, 2 to 30640
Data columns (total 14 columns):
 #   Column                 Non-Null Count  Dtype 
---  ------                 --------------  ----- 
 0   gender                 21721 non-null  object
 1   ethnic_group           21721 non-null  object
 2   parents_education      21721 non-null  object
 3   lunch_type             21721 non-null  object
 4   test_preparation       21721 non-null  object
 5   parent_marital_status  21721 non-null  object
 6   practice_sports        21721 non-null  object
 7   is_first_child         21721 non-null  object
 8   number_of_siblings     21721 non-null  Int64 
 9   transport_means        21721 non-null  object
 10  weekly_study_hours     21721 non-null  object
 11  math_score             21721 non-null  int64 
 12  reading_score          21721 non-null  int64 
 13  writing_score          21721 non-null  int64 
dtypes: Int64(1), int64(3), object(10)
memory usage: 2.5+ MB
```

#### **Compare the descriptive statistics of the two datasets**
```
descriptive_stats = pd.concat([df.describe(), df1b.describe()], axis=1, keys=['Dataset with Removed Null Values','Dataset with Mode Imputation'])

descriptive_stats
```
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Dataset with Removed Null Values</th>
      <th colspan="4" halign="left">Dataset with Mode Imputation</th>
    </tr>
    <tr>
      <th></th>
      <th>number_of_siblings</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
      <th>number_of_siblings</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>29069.0</td>
      <td>30641.000000</td>
      <td>30641.000000</td>
      <td>30641.000000</td>
      <td>21721.0</td>
      <td>21721.000000</td>
      <td>21721.000000</td>
      <td>21721.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.145894</td>
      <td>66.558402</td>
      <td>69.377533</td>
      <td>68.418622</td>
      <td>2.140785</td>
      <td>66.589844</td>
      <td>69.467152</td>
      <td>68.534736</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.458242</td>
      <td>15.361616</td>
      <td>14.758952</td>
      <td>15.443525</td>
      <td>1.447413</td>
      <td>15.382437</td>
      <td>14.787346</td>
      <td>15.488860</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>56.000000</td>
      <td>59.000000</td>
      <td>58.000000</td>
      <td>1.0</td>
      <td>56.000000</td>
      <td>59.000000</td>
      <td>58.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.0</td>
      <td>67.000000</td>
      <td>70.000000</td>
      <td>69.000000</td>
      <td>2.0</td>
      <td>67.000000</td>
      <td>70.000000</td>
      <td>69.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.0</td>
      <td>78.000000</td>
      <td>80.000000</td>
      <td>79.000000</td>
      <td>3.0</td>
      <td>78.000000</td>
      <td>80.000000</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>7.0</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>

Based on the descriptive statistics from the resulting datasets from the two methods, the statistic values for the dataset with mode imputations applied varies from the dataset with all null values removed by a very small amount (less than 1.00).

With this result, we will proceed on using the dataset obtained from mode imputation as it contains more data compared to the dataset with all null values removed.

### **EXPLORATORY DATA ANALYSIS (EDA)**

#### **1. Gender**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='gender')

tag = sns.countplot(data=df1b, x='gender')

for i in tag.patches:
    tag.annotate(f'{i.get_height()}', (i.get_x() + i.get_width() / 2., i.get_height() / 2), ha='center', va='bottom')

plt.show()

df1b.groupby(['gender'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/587a0054-8d35-4997-8712-f6628510896e)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>64.162861</td>
      <td>64.0</td>
      <td>72.968247</td>
      <td>73.0</td>
      <td>72.980893</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>male</th>
      <td>69.075862</td>
      <td>69.0</td>
      <td>65.880895</td>
      <td>66.0</td>
      <td>63.980429</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>

  - There are more female than male students. The difference is quite small as the female students are only 261 (1.20%) more than the male students.
    
  - Male students generally score higher in Math, while female students generally score higher in Reading and Writing.

#### **2. Ethnic Group**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='ethnic_group')

plt.show()

df1b.groupby(['ethnic_group'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/5be74153-ad35-4d2b-a646-e451b1264115)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>ethnic_group</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>group A</th>
      <td>62.974699</td>
      <td>64.0</td>
      <td>66.709639</td>
      <td>66.0</td>
      <td>65.120482</td>
      <td>65.0</td>
    </tr>
    <tr>
      <th>group B</th>
      <td>63.536916</td>
      <td>64.0</td>
      <td>67.347934</td>
      <td>68.0</td>
      <td>65.994581</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>group C</th>
      <td>64.735345</td>
      <td>65.0</td>
      <td>68.518391</td>
      <td>69.0</td>
      <td>67.122845</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>group D</th>
      <td>67.669339</td>
      <td>68.0</td>
      <td>70.515945</td>
      <td>71.0</td>
      <td>71.032425</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>group E</th>
      <td>75.210526</td>
      <td>76.0</td>
      <td>74.266100</td>
      <td>75.0</td>
      <td>72.694671</td>
      <td>73.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students from Ethnic Group E have the highest scores from the 3 subjects.
    
  - The hierarchy of students based on exam scores is the same from all the 3 subjects - 1. group E, 2. group D, 3. group C, 4. group B, 5. group A (from highest to lowest). This is true for both the mean and median scores.
    
  - There seems to be a clear correlation between the ethnic group and the exam scores.

#### **3. Parent's Educational Background**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='parents_education')
plt.xticks(rotation=90)

plt.show()

df1b.groupby(['parents_education'])[['math_score','writing_score','reading_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/b353b7cf-3df1-4817-b9ab-9664fc194880)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">writing_score</th>
      <th colspan="2" halign="left">reading_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>parents_education</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>associate's degree</th>
      <td>68.477651</td>
      <td>69.0</td>
      <td>70.403947</td>
      <td>71.0</td>
      <td>71.159534</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>bachelor's degree</th>
      <td>70.691892</td>
      <td>71.0</td>
      <td>73.737452</td>
      <td>74.0</td>
      <td>73.440154</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>high school</th>
      <td>64.262652</td>
      <td>64.0</td>
      <td>65.410965</td>
      <td>66.0</td>
      <td>67.183693</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>master's degree</th>
      <td>72.284500</td>
      <td>73.0</td>
      <td>76.335513</td>
      <td>77.0</td>
      <td>75.785481</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>some college</th>
      <td>66.507113</td>
      <td>66.0</td>
      <td>68.603086</td>
      <td>69.0</td>
      <td>69.285314</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>some high school</th>
      <td>62.498429</td>
      <td>63.0</td>
      <td>63.634276</td>
      <td>64.0</td>
      <td>65.499154</td>
      <td>65.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who have parents that obtained Master's Degree scored the highest in all three exams.
    
  - The hierarchy of students based on exam scores is the same for all the 3 subjects- 1. master's degree, 2. bachelor's degree, 3. associate's degree, 4. some college, 5. some highschool (from highest to lowest). This is true for both the mean and median scores.
    
  - There is a correlation between the exam scores of the students and their parent's highest educational attainment. This is because students with parents who have a degree scored higher than students who have parents without a degree.
    
#### **4. Lunch Type**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='lunch_type')

plt.show()

df1b.groupby(['lunch_type'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/fd042afb-83e9-4413-8f08-0224689d4fb7)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>lunch_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>free/reduced</th>
      <td>58.879832</td>
      <td>59.0</td>
      <td>64.245577</td>
      <td>64.0</td>
      <td>62.746298</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>standard</th>
      <td>70.765507</td>
      <td>71.0</td>
      <td>72.295103</td>
      <td>73.0</td>
      <td>71.669695</td>
      <td>72.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who have standard lunch type scored higher than the students with free/reduced lunch type. The difference is quite noticeable especially for the Math exam.
    
  - There are more students that have standard lunch type.
    
  - The type of lunch a student has seems to affect their exam scores for the 3 subjects. It seems like student who enjoyed better food tend to perform better in exams.
    
#### **5. Test Preparation**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='test_preparation')

plt.show()

df1b.groupby(['test_preparation'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/aa52c594-2d07-4374-b8b5-5a2f6cdf8d20)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>test_preparation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>completed</th>
      <td>69.681842</td>
      <td>70.0</td>
      <td>73.863950</td>
      <td>74.0</td>
      <td>74.860499</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>none</th>
      <td>64.947840</td>
      <td>65.0</td>
      <td>67.132234</td>
      <td>67.0</td>
      <td>65.175442</td>
      <td>65.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who prepared for the exam better for all the subjects compared to students with no preparation.

  - However, there are more students who do not prepare for the exams, as shown in the plot.

#### **6. Parent's Marital Status**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='parent_marital_status')

plt.show()

df1b.groupby(['parent_marital_status'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/1e688795-ca1e-43d2-abfa-c48c979a4b62)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>parent_marital_status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>divorced</th>
      <td>66.652371</td>
      <td>67.0</td>
      <td>69.700085</td>
      <td>70.0</td>
      <td>68.879012</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>married</th>
      <td>66.666536</td>
      <td>67.0</td>
      <td>69.429912</td>
      <td>70.0</td>
      <td>68.447669</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>single</th>
      <td>66.246203</td>
      <td>66.0</td>
      <td>69.320344</td>
      <td>70.0</td>
      <td>68.446243</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>widowed</th>
      <td>67.849515</td>
      <td>69.0</td>
      <td>70.415049</td>
      <td>71.0</td>
      <td>69.368932</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who have widowed parents scored better that the other students.

  - The difference in scores for all the variables is very small. There seems to be little to no correlation between the exam scores and the marital status of the student's parents.

#### **7. Practice Sports**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='practice_sports')

plt.show()

df1b.groupby(['practice_sports'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/20ece306-e2a9-4834-b215-bfefca3ff0ea)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>practice_sports</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>never</th>
      <td>64.487848</td>
      <td>65.0</td>
      <td>68.651638</td>
      <td>69.0</td>
      <td>66.876717</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>regularly</th>
      <td>67.677549</td>
      <td>68.0</td>
      <td>69.946020</td>
      <td>70.0</td>
      <td>69.638918</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>sometimes</th>
      <td>66.375916</td>
      <td>66.0</td>
      <td>69.345096</td>
      <td>70.0</td>
      <td>68.196891</td>
      <td>68.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who regularly practice sports scored higher than other students.

  - The difference between the mean and median scores for students who regularly practice sports and those who sometime practice it is very small. However, there is quite a noticeable difference from the results of the two variables compared to the exam scores obtained from students who never practice sports. This findings is true for Math and Writing score, but not for the Reading score.

  - There seems to be a little correlation between the exam scores and whether a student practice sports or not. This is true for Math and Writing scores.

#### **8. Is First Child?**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='is_first_child')

plt.show()

df1b.groupby(['is_first_child'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/7f55d075-af89-4ab4-83ce-e72a174ccd2f)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>is_first_child</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>no</th>
      <td>66.418924</td>
      <td>67.0</td>
      <td>69.336848</td>
      <td>70.0</td>
      <td>68.483785</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>yes</th>
      <td>66.679857</td>
      <td>67.0</td>
      <td>69.535775</td>
      <td>70.0</td>
      <td>68.561569</td>
      <td>69.0</td>
    </tr>
  </tbody>
</table>
</div>

  - The exam scores for all three subjects between students who are first child and those who are not is quite similar. This means that whether a student is a first child or not does not contribute to their exam scores.

#### **9. Number of Siblings**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='number_of_siblings')

plt.show()

df1b.groupby(['number_of_siblings'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/08fc703e-fd81-4080-aa50-59f38e630d78)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>number_of_siblings</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>66.673744</td>
      <td>67.0</td>
      <td>69.643847</td>
      <td>70.0</td>
      <td>68.766031</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>66.474546</td>
      <td>67.0</td>
      <td>69.331806</td>
      <td>70.0</td>
      <td>68.337698</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66.568543</td>
      <td>67.0</td>
      <td>69.459698</td>
      <td>70.0</td>
      <td>68.540498</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>66.717838</td>
      <td>67.0</td>
      <td>69.512649</td>
      <td>70.0</td>
      <td>68.684757</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66.341353</td>
      <td>66.0</td>
      <td>69.462097</td>
      <td>70.0</td>
      <td>68.474807</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>66.968880</td>
      <td>67.0</td>
      <td>69.680498</td>
      <td>70.0</td>
      <td>68.557054</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>66.717073</td>
      <td>66.0</td>
      <td>68.726829</td>
      <td>70.0</td>
      <td>67.643902</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>67.471154</td>
      <td>69.0</td>
      <td>70.447115</td>
      <td>71.0</td>
      <td>69.687500</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>

  - Whether a students have more siblings or not does not affect their exam scores, as shown in the aggregate results. The mean and median of the exam results have very little difference between each variables.

#### **10. Transport Means**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='transport_means')

plt.show()

df1b.groupby(['transport_means'])[['math_score','reading_score','writing_score']].agg(['mean','median'])
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/d05b0e16-6385-4727-a83b-31adffe5d07c)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">math_score</th>
      <th colspan="2" halign="left">reading_score</th>
      <th colspan="2" halign="left">writing_score</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
      <th>mean</th>
      <th>median</th>
    </tr>
    <tr>
      <th>transport_means</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>private</th>
      <td>66.500947</td>
      <td>67.0</td>
      <td>69.469401</td>
      <td>70.0</td>
      <td>68.530487</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>school_bus</th>
      <td>66.652392</td>
      <td>67.0</td>
      <td>69.465569</td>
      <td>70.0</td>
      <td>68.537725</td>
      <td>69.0</td>
    </tr>
  </tbody>
</table>
</div>

  - There is also no correlation between the exam scores and the student's means of transportation. The mean and median scores for all the three exams scores are almost similar for each variable.

#### **11. Weekly Study Hours**
```
plt.subplots(1,1, figsize=(5,3))

sns.countplot(data=df1b, x='weekly_study_hours')

plt.show()

df1b.groupby(['weekly_study_hours'])[['math_score','reading_score','writing_score']].agg('mean','median')
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/d255111c-cf64-4088-8716-f4430b373a89)

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
    <tr>
      <th>weekly_study_hours</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5 - 10</th>
      <td>66.904703</td>
      <td>69.736183</td>
      <td>68.760728</td>
    </tr>
    <tr>
      <th>&lt; 5</th>
      <td>64.601423</td>
      <td>68.323674</td>
      <td>67.255042</td>
    </tr>
    <tr>
      <th>&gt; 10</th>
      <td>68.740147</td>
      <td>70.413428</td>
      <td>69.841533</td>
    </tr>
  </tbody>
</table>
</div>

  - Students who studied more than 10 hours per week scored higher for all three exams.

  - There is a correlation between exam scores and the weekly study hours of a student, as students who spent more time studying tend to get higher scores, and student who spent the least time studying scored the lowest.

  - As shown in the plot, most students study for 5 to 10 hours per week.
Before proceeding with the Model Building, we will exclude the variables that has no correlation with the exam scores. These variables are is_first_child, number_of_siblings, and transport_means.
```
## Drop 'is_first_child', 'number_of_siblings', and 'transport_means' columns

df1b = df1b.drop(columns=['is_first_child','number_of_siblings','transport_means'], axis=1)
```

### **MODEL BUILDING**

#### **XGBoost Model**
```
## Create a copy of the dataset to be encoded

df1b_enc = df1b.copy()

df1b_enc.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ethnic_group</th>
      <th>parents_education</th>
      <th>lunch_type</th>
      <th>test_preparation</th>
      <th>parent_marital_status</th>
      <th>practice_sports</th>
      <th>weekly_study_hours</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>single</td>
      <td>sometimes</td>
      <td>&lt; 5</td>
      <td>87</td>
      <td>93</td>
      <td>91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>sometimes</td>
      <td>5 - 10</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
    <tr>
      <th>5</th>
      <td>female</td>
      <td>group B</td>
      <td>associate's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>married</td>
      <td>regularly</td>
      <td>5 - 10</td>
      <td>73</td>
      <td>84</td>
      <td>79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>female</td>
      <td>group B</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>widowed</td>
      <td>never</td>
      <td>5 - 10</td>
      <td>85</td>
      <td>93</td>
      <td>89</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>group B</td>
      <td>some college</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>married</td>
      <td>sometimes</td>
      <td>&gt; 10</td>
      <td>41</td>
      <td>43</td>
      <td>39</td>
    </tr>
  </tbody>
</table>
</div>
For the new dataset, we will encode categorical variables.

We will use dummy encode the following columns:

  - ethnic_group, parents_education, parent_marital_status, practice_sports, and weekly_study_hours

And convert the following columns into binary:

  - gender, lunch_type, and test_preparation
```
## Define variables to dummy encode

columns_to_encode = ['ethnic_group','parents_education','parent_marital_status','practice_sports','weekly_study_hours']

for col in columns_to_encode:
    df1b_enc = pd.get_dummies(data=df1b_enc, columns=[col], drop_first=True, dtype=int)
    
## Convert selected variables into binary
    
df1b_enc['gender'] = np.where(df1b_enc['gender']=='male',1,0)
df1b_enc['lunch_type'] = np.where(df1b_enc['lunch_type']=='standard',1,0)
df1b_enc['test_preparation'] = np.where(df1b_enc['test_preparation']=='completed',1,0)

## Replace characters that could cause errors

df1b_enc.columns = df1b_enc.columns.str.replace(' ', '_').str.replace("'","").str.replace('>','_more_than_').str.replace('<','_less_than_')
    
df1b_enc.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>lunch_type</th>
      <th>test_preparation</th>
      <th>math_score</th>
      <th>reading_score</th>
      <th>writing_score</th>
      <th>ethnic_group_group_B</th>
      <th>ethnic_group_group_C</th>
      <th>ethnic_group_group_D</th>
      <th>ethnic_group_group_E</th>
      <th>parents_education_bachelors_degree</th>
      <th>parents_education_high_school</th>
      <th>parents_education_masters_degree</th>
      <th>parents_education_some_college</th>
      <th>parents_education_some_high_school</th>
      <th>parent_marital_status_married</th>
      <th>parent_marital_status_single</th>
      <th>parent_marital_status_widowed</th>
      <th>practice_sports_regularly</th>
      <th>practice_sports_sometimes</th>
      <th>weekly_study_hours__less_than__5</th>
      <th>weekly_study_hours__more_than__10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>87</td>
      <td>93</td>
      <td>91</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>73</td>
      <td>84</td>
      <td>79</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>85</td>
      <td>93</td>
      <td>89</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>41</td>
      <td>43</td>
      <td>39</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

#### **Split the data**
```
## Define the target variable

target_var = df1b_enc[['math_score','reading_score','writing_score']]

target_math = target_var['math_score']
target_reading = target_var['reading_score']
target_writing = target_var['writing_score']

## Define the predictor variable

predictor_var = df1b_enc.drop(['math_score','reading_score','writing_score'], axis=1)

## Split into train and test sets

X_train, X_test, y_math_train, y_math_test, y_reading_train, y_reading_test, y_writing_train, y_writing_test = train_test_split(predictor_var, target_math, 
                                                                                                                                target_reading, target_writing,
                                                                                                                                test_size=0.25, random_state=42)
```
#### **Cross-validated hyperparameter tuning**

```
## Create XGBoost models for each test scores

xgb_math = XGBRegressor()
xgb_reading = XGBRegressor()
xgb_writing = XGBRegressor()

## Hyperparameter tuning with GridSearch

params = {'max_depth': [4,5,6,7,8], 
          'min_child_weight': [1,2,3,4,5],
          'learning_rate': [0.1, 0.2, 0.3],
          'n_estimators': [75, 100, 125]
          }

scores = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)

grid_math = GridSearchCV(xgb_math, param_grid=params, cv=5, scoring=scores)
grid_reading = GridSearchCV(xgb_reading, param_grid=params, cv=5, scoring=scores)
grid_writing = GridSearchCV(xgb_writing, param_grid=params, cv=5, scoring=scores)

grid_math.fit(X_train, y_math_train)
grid_reading.fit(X_train, y_reading_train)
grid_writing.fit(X_train, y_reading_train)

## Obtain the best hyperparameters

math_best_params = grid_math.best_params_
reading_best_params = grid_reading.best_params_
writing_best_params = grid_writing.best_params_
```
```
## Perform cross-validation and get predicted values

math_preds = cross_val_predict(grid_math.best_estimator_, X_train, y_math_train, cv=5)
reading_preds = cross_val_predict(grid_reading.best_estimator_, X_train, y_math_train, cv=5)
writing_preds = cross_val_predict(grid_writing.best_estimator_, X_train, y_math_train, cv=5)
```
```
## Print RMSE values

rmse_math = np.sqrt(mean_squared_error(y_math_train, math_preds))
rmse_reading = np.sqrt(mean_squared_error(y_reading_train, reading_preds))
rmse_writing = np.sqrt(mean_squared_error(y_writing_train, writing_preds))

print('RMSE Value for Math Scores:', rmse_math)
print('RMSE Value for Reading Scores:', rmse_reading)
print('RMSE Value for Writing Scores:', rmse_writing)
```
RMSE Value for Math Scores: 12.994796880123351
RMSE Value for Reading Scores: 14.740659354449292
RMSE Value for Writing Scores: 14.90092506353194
```
## Create a scatterplot to visualize actual versus predicted scores

test_scores = {'Math':(y_math_train, math_preds),
               'Reading':(y_reading_train, reading_preds),
               'Writing':(y_writing_train, writing_preds)}

plt.figure(figsize=(18,6))

for i, (subj_score_name,(actual_score, predicted_score)) in enumerate(test_scores.items()):
    plt.subplot(1,3, i+1)
    plt.scatter(actual_score, predicted_score, s=8, alpha=0.5)
    plt.title(f'{subj_score_name} Scores')
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    lr = LinearRegression()
    lr.fit(actual_score.values.reshape(-1,1), predicted_score)
    plt.plot(actual_score, lr.predict(actual_score.values.reshape(-1,1)), color='red', linewidth=2)
    
plt.tight_layout()
plt.show()
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/e8f5a8f1-c394-4ba6-ae5b-55227b68f8d6)

The **Root Mean Square Error (RMSE)** for the three exam scores are the following:

- Math Scores: 13.000
  
- Reading Scores: 14.744
  
- Writing Scores: 14.904

Which means that the predicted scores for each subject deviate from the actual scores by the RMSE value of that respective subject. It means that the predicted values for Math Scores are off by an average of 13.000 points from the actual values, 14.744 points for Reading Scores, and 14.984 points for Writing Scores.

These are high values of errors considering we only have values of 0 to 100 for each exam scores.

The results is also backed up by the scatterplots for each subjects, which shows that many points are scattered far from the best fit line. Which indicates a weak relationship between variables.
```
## Create a table showing comparison between actual and predicted values of test results

comparison_table = pd.DataFrame({'Actual Math Score': y_math_train.values,
                                 'Predicted Math Score': math_preds,
                                 'Actual Reading Score': y_reading_train.values,
                                 'Predicted Reading Score': reading_preds,
                                 'Actual Writing Score': y_writing_train.values,
                                 'Predicted Writing Score': writing_preds
                                 })

comparison_table.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual Math Score</th>
      <th>Predicted Math Score</th>
      <th>Actual Reading Score</th>
      <th>Predicted Reading Score</th>
      <th>Actual Writing Score</th>
      <th>Predicted Writing Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69</td>
      <td>70.787338</td>
      <td>69</td>
      <td>70.787338</td>
      <td>70</td>
      <td>70.787338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>69.141335</td>
      <td>80</td>
      <td>69.141335</td>
      <td>81</td>
      <td>69.141335</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79</td>
      <td>77.691803</td>
      <td>78</td>
      <td>77.691803</td>
      <td>70</td>
      <td>77.691803</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
      <td>68.424431</td>
      <td>58</td>
      <td>68.424431</td>
      <td>60</td>
      <td>68.424431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>59.363270</td>
      <td>49</td>
      <td>59.363270</td>
      <td>53</td>
      <td>59.363270</td>
    </tr>
  </tbody>
</table>
</div>

The predicted values showed huge variance from the actual scores for all subjects.

#### **Feature Importance**
```
## Fit the models for each scores

xgb_math.fit(predictor_var, target_math)
xgb_reading.fit(predictor_var, target_reading)
xgb_writing.fit(predictor_var, target_writing)

## Plot feature importance for each scores

fig, ax = plt.subplots(1,3, figsize=(21,5))

xgb.plot_importance(xgb_math, importance_type='weight', title='Feature Importance for Math Scores', ax=ax[0])
xgb.plot_importance(xgb_reading, importance_type='weight', title='Feature Importance for Reading Scores', ax=ax[1])
xgb.plot_importance(xgb_writing, importance_type='weight', title='Feature Importance for Writing Scores', ax=ax[2])

plt.tight_layout()
plt.show()
```
![image](https://github.com/ryanlacsamana/Exam_Scores_Prediction/assets/138304188/815ad211-f483-42de-87ab-4a1178799f49)

The three exam scores showed different feature importances. For the Math Scores, it seems like test preparation, gender, and number of study hours have the most weight. For the Reading Scores, gender, test preparation, and whether a student practice sports or not played the biggest role. For the Writing Scores, the lunch type, gender, and test preparation have the most weight.

### **CONCLUSION**

  - The model could use a lot of improvement in prediction as it returned a high value of error.

  - There are variables that appear the most in the feature importance, these variables are test_preparation, gender, and lunch_type, weekly_study_hours, and practice_sports. The variable test_preparation could be an obvious variable to appear on the top of the feature importance. However, variables such as gender appeared on the top 3 feature importance for the three exams, which is quite questionable.

  - In order to increase the model's performance, more sample data should be provided to increase the predictive ability of the model.

