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
