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
