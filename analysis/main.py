import pandas as pd
import pylab as pl
import numpy as np
import re

##importing and formatting your data

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


df = pd.read_csv("./data/credit-training.csv", index_col=0)

# column names are in camelCase; easier to read in snake_case
df.head()

# convert each column name to snake case
df.columns = [camel_to_snake(col) for col in df.columns]

##handling nulls

# what we really want to do is figure out which variables have
# null values and handle them accordingly. to do this, we're going to
# 'melt' our data into 'long' format'
# the end goal is to have a table that tells us for each variable, how many
# instances are null vs. non-null
df_lng = pd.melt(df)

# now our data is a series of (key, value) rows. think of when you've done
# this in Excel so that you can create a pivot table 
df_lng.head()

null_variables = df_lng.value.isnull()

# crosstab creates a frequency table between 2 variables
# it's going to automatically enumerate the possibilities between
# the two Series and show you a count of occurrences in each possible bucket
print pd.crosstab(df_lng.variable, null_variables)

# let's abstract that code into a function so we can easily recalculate it
def print_null_freq(df):
    """
    for a given DataFrame, calculates how many values for each variable is null
    and prints the resulting table to stdout
    """
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    print pd.crosstab(df_lng.variable, null_variables)

# you can see that monthly_income and number_of_dependents both had null values
# we need to come up with a strategy for replacing null values so that we can 
# still use these records in our analysis.

# for number_of_dependents let's keep things simple and intuitive. if someone
# didn't specify how many dependents they had then let's assume it's becasue they
# don't have any to begin with.
df.number_of_dependents = df.number_of_dependents.fillna(0)
# proof that the number_of_dependents no longer contains nulls
print_null_freq(df)

#
df.monthly_income.describe()
replacement_value = df.monthly_income.median()
replacement_value = df.monthly_income.mean()

from sklearn.neighbors import KNeighborsRegressor

income_imputer = KNeighborsRegressor(n_neighbors=1)
nonnull_data= df[df.monthly_income.isnull()==False]
null_data = df[df.monthly_income.isnull()==True]
X = nonnull_data[['debt_ratio', 'age']]
y = nonnull_data.monthly_income
income_imputer.fit(X, y)

# calculate the imputation and then remove the null values from null_data
new_values = income_imputer.predict(null_data[['debt_ratio', 'age']])
null_data.monthly_income = new_values

# reconstruct df by combining nonnull and null data
df = nonnull_data.append(null_data)
# no more nulls!
print_null_freq(df)



##feature selection

features = np.array(['revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse',
            'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans', 'number_of_times90_days_late',
            'number_real_estate_loans_or_lines','number_of_time60-89_days_past_due_not_worse', 'number_of_dependents'])

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df[features], df.serious_dlqin2yrs, test_size=0.25)

clf = RandomForestClassifier(compute_importances=True)
clf.fit(train_x, train_y)

# from the calculated importances, order them from most to least important
# and make a barplot so we can visualize what is/isn't important
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)

padding = np.arange(len(features)) + 0.5
pl.barh(padding, importances[sorted_idx], align='center')
pl.yticks(padding, features[sorted_idx])
pl.xlabel("Relative Importance")
pl.title("Variable Importance")
pl.show()

## exploring features
age_means = df[['age', 'serious_dlqin2yrs']].groupby("age").mean()
age_means.plot()

df["age_bucket"] = pd.cut(df.age, range(-1, 110, 10))
buckets = [-1, 25] + range(25, 80, 5) + [80, 120]
df["age_bucket"] = pd.cut(df.age, buckets) 
pd.crosstab(df.age_bucket, df.serious_dlqin2yrs)
df[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean()
df[["age_bucket", "serious_dlqin2yrs"]].groupby("age_bucket").mean().plot()



##building your model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

#we're going to use the simplest type of cross validation. we'll simply split our data into 2 groups:
#training and test. we'll use the training set to calibrate our model and then use the test set to 
#evaluate how effective it is.
is_test = np.random.uniform(0, 1, len(df)) > 0.75
train = df[is_test==True]
test = df[is_test==False]

clf = KNeighborsClassifier(n_neighbors=5)
clf = RandomForestClassifier()
clf = GradientBoostingClassifier()

features = ['debt_ratio', 'age', 'number_of_dependents', 'number_of_times90_days_late']
clf.fit(train[features], train['serious_dlqin2yrs'])

##evaluating with an ROC curve (http://scikit-learn.org/stable/auto_examples/plot_roc.html#example-plot-roc-py)
from sklearn.metrics import roc_curve, auc
preds = clf.predict_proba(test[features])
fpr, tpr, thresholds = roc_curve(test['serious_dlqin2yrs'], preds[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()


#Converting to credit score
# we're going to take the P(delinquent) outputted by the model and convert it to a FICO style score
# we calculate the log odds which we then convert into 'points'. in this case, a increase/decrease
# in 40 points (arbritrary) means a person's riskness has halved/doubled--40/log(2). we're starting with a base
# score of 340 (arbitrary).
p = preds[::, 1]
odds = (1 - p) / p
score = np.log(odds)*(40/np.log(2)) + 340
pl.hist(score)

def convert_prob_to_score(p):
    """
    takes a probability and converts it to a score
    Example:
        convert_prob_to_score(0.1)
        > 340
    """
    odds = (1 - p) / p
    return np.log(odds)*(40/np.log(2)) + 340



##Deploying to Yhat
from yhat import BaseModel, Yhat


yh = Yhat("greg", "abcd1234")

class LoanModel(BaseModel):
    def transform(self, newdata):
        df = pd.DataFrame(newdata)
        # handle nulls here
        # df['monthly_income'] = self.income_imputer.predict(df[[]])
        df['number_of_dependents'] = df['number_of_dependents'].fillna(0)
        return df

    def predict(self, df):
        data = df[self.features]
        result = {}
        p = self.clf.predict_proba(data)
        p = p[::, 1]
        score = convert_prob_to_score(p)
        result["prob"] = p
        result["score"] = score
        return result


loan_model = LoanModel(clf=clf, features=features, income_imputer=income_imputer,
                 udfs=[convert_prob_to_score])




