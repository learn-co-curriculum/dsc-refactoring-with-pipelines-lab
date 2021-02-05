# Refactoring Your Code to Use Pipelines - Lab

## Introduction

In this lab, you will practice refactoring existing scikit-learn code into code that uses pipelines.

## Objectives

You will be able to:

* Practice reading and interpreting existing scikit-learn preprocessing code
* Think logically about how to organize steps with `Pipeline`, `ColumnTransformer`, and `FeatureUnion`
* Refactor existing preprocessing code into a pipeline

## Ames Housing Data Preprocessing Steps

In this lesson we will return to the Ames Housing dataset to perform some familiar preprocessing steps, then add an `ElasticNet` model as the estimator.

#### 1. Drop Irrelevant Columns

For the purposes of this lab, we will only be using a subset of all of the features present in the Ames Housing dataset. In this step you will drop all irrelevant columns.

#### 2. Handle Missing Values

Often for reasons outside of a data scientist's control, datasets are missing some values. In this step you will assess the presence of NaN values in our subset of data, and use `MissingIndicator` and `SimpleImputer` from the `sklearn.impute` submodule to handle any missing values.

#### 3. Convert Categorical Features into Numbers

A built-in assumption of the scikit-learn library is that all data being fed into a machine learning model is already in a numeric format, otherwise you will get a `ValueError` when you try to fit a model. In this step you will `OneHotEncoder`s to replace columns containing categories with "dummy" columns containing 0s and 1s.

#### 4. Add Interaction Terms

This step gets into the feature engineering part of preprocessing. Does our model improve as we add interaction terms? In this step you will use a `PolynomialFeatures` transformer to augment the existing features of the dataset.

#### 5. Scale Data

Because we are using a model with regularization, it's important to scale the data so that coefficients are not artificially penalized based on the units of the original feature. In this step you will use a `StandardScaler` to standardize the units of your data.

## Getting the Data

The cell below loads the Ames Housing data into the relevant train and test data, split into features and target.


```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read in the data and separate into X and y
df = pd.read_csv("data/ames.csv")
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

## Original Preprocessing Code

The following code uses scikit-learn to complete all of the above steps, outside of the context of a pipeline. It is broken down into functions for improved readability.

*Step 1: Drop Irrelevant Columns*


```python
def drop_irrelevant_columns(X):
    relevant_columns = [
        'LotFrontage',  # Linear feet of street connected to property
        'LotArea',      # Lot size in square feet
        'Street',       # Type of road access to property
        'OverallQual',  # Rates the overall material and finish of the house
        'OverallCond',  # Rates the overall condition of the house
        'YearBuilt',    # Original construction date
        'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
        'GrLivArea',    # Above grade (ground) living area square feet
        'FullBath',     # Full bathrooms above grade
        'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
        'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
        'Fireplaces',   # Number of fireplaces
        'FireplaceQu',  # Fireplace quality
        'MoSold',       # Month Sold (MM)
        'YrSold'        # Year Sold (YYYY)
    ]
    return X.loc[:, relevant_columns]
```

*Step 2: Handle Missing Values*


```python

from sklearn.impute import MissingIndicator, SimpleImputer

def handle_missing_values(X):
    # Replace fireplace quality NaNs with "N/A"
    X["FireplaceQu"] = X["FireplaceQu"].fillna("N/A")
    
    # Missing indicator for lot frontage
    frontage = X[["LotFrontage"]]
    missing_indicator = MissingIndicator()
    frontage_missing = missing_indicator.fit_transform(frontage)
    X["LotFrontage_Missing"] = frontage_missing
    
    # Imputing missing values for lot frontage
    imputer = SimpleImputer(strategy="median")
    frontage_imputed = imputer.fit_transform(frontage)
    X["LotFrontage"] = frontage_imputed
    
    return X, [missing_indicator, imputer]
```

*Step 3: Convert Categorical Features into Numbers*


```python

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def handle_categorical_data(X):
    # Binarize street
    street = X["Street"]
    binarizer_street = LabelBinarizer()
    street_binarized = binarizer_street.fit_transform(street)
    X["Street"] = street_binarized
    
    # Binarize frontage missing
    frontage_missing = X["LotFrontage_Missing"]
    binarizer_frontage_missing = LabelBinarizer()
    frontage_missing_binarized = binarizer_frontage_missing.fit_transform(frontage_missing)
    X["LotFrontage_Missing"] = frontage_missing_binarized
    
    # One-hot encode fireplace quality
    fireplace_quality = X[["FireplaceQu"]]
    ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
    fireplace_quality_encoded = ohe.fit_transform(fireplace_quality)
    fireplace_quality_encoded = pd.DataFrame(
        fireplace_quality_encoded,
        columns=ohe.categories_[0],
        index=X.index
    )
    X.drop("FireplaceQu", axis=1, inplace=True)
    X = pd.concat([X, fireplace_quality_encoded], axis=1)
    
    return X, [binarizer_street, binarizer_frontage_missing, ohe]
```

*Step 4: Add Interaction Terms*


```python

from sklearn.preprocessing import PolynomialFeatures

def add_interaction_terms(X):
    poly_column_names = [
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "YearBuilt",
        "GrLivArea"
    ]
    poly_columns = X[poly_column_names]
    
    # Generate interaction terms
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    poly_columns_expanded = poly.fit_transform(poly_columns)
    poly_columns_expanded = pd.DataFrame(
        poly_columns_expanded,
        columns=poly.get_feature_names(poly_column_names),
        index=X.index
    )
    
    # Replace original columns with expanded columns
    # including interaction terms
    X.drop(poly_column_names, axis=1, inplace=True)
    X = pd.concat([X, poly_columns_expanded], axis=1)
    return X, [poly]

```

*Step 5: Scale Data*


```python

from sklearn.preprocessing import StandardScaler

def scale(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(
        X_scaled,
        columns=X.columns,
        index=X.index
    )
    return X, [scaler]
```

In the cell below, we execute all of the above steps on the training data:


```python

from sklearn.linear_model import ElasticNet

X_train = drop_irrelevant_columns(X_train)
X_train, step_2_transformers = handle_missing_values(X_train)
X_train, step_3_transformers = handle_categorical_data(X_train)
X_train, step_4_transformers = add_interaction_terms(X_train)
X_train, step_5_transformers = scale(X_train)

model = ElasticNet(random_state=1)
model.fit(X_train, y_train)
```




    ElasticNet(random_state=1)



(The transformers have all been returned by the functions, so theoretically we could use them to transform the test data appropriately, but for now we'll skip that step for time.)

## Refactoring into a Pipeline

Great, now let's refactor that into pipeline code! Some of the following code has been completed for you, whereas other code you will need to fill in.

First we'll reset the values of our data:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

### 1. Drop Irrelevant Columns

Previously, we just used pandas dataframe slicing to select the relevant elements. Now we'll need something a bit more complicated.

When using `ColumnTransformer`, the default behavior is to drop irrelevant columns anyway. So what we really need now is to break down the full set of columns into:

* Columns that should "pass through" without any changes made
* Columns that require preprocessing
* Columns we don't want

Luckily we don't actually need a list of the third category, since they will be dropped by default.

In the cell below, we create the necessary lists for you:


```python

relevant_columns = [
    'LotFrontage',  # Linear feet of street connected to property
    'LotArea',      # Lot size in square feet
    'Street',       # Type of road access to property
    'OverallQual',  # Rates the overall material and finish of the house
    'OverallCond',  # Rates the overall condition of the house
    'YearBuilt',    # Original construction date
    'YearRemodAdd', # Remodel date (same as construction date if no remodeling or additions)
    'GrLivArea',    # Above grade (ground) living area square feet
    'FullBath',     # Full bathrooms above grade
    'BedroomAbvGr', # Bedrooms above grade (does NOT include basement bedrooms)
    'TotRmsAbvGrd', # Total rooms above grade (does not include bathrooms)
    'Fireplaces',   # Number of fireplaces
    'FireplaceQu',  # Fireplace quality
    'MoSold',       # Month Sold (MM)
    'YrSold'        # Year Sold (YYYY)
]

poly_column_names = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "GrLivArea"
]

# Use set logic to combine lists without overlaps while maintaining order
columns_needing_preprocessing = ["FireplaceQu", "LotFrontage", "Street"] \
    + list(set(poly_column_names) - set(["LotFrontage"]))
passthrough_columns = list(set(relevant_columns) - set(columns_needing_preprocessing))

print("Need preprocessing:", columns_needing_preprocessing)
print("Passthrough:", passthrough_columns)
```

    Need preprocessing: ['FireplaceQu', 'LotFrontage', 'Street', 'GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']
    Passthrough: ['BedroomAbvGr', 'YrSold', 'MoSold', 'FullBath', 'Fireplaces', 'YearRemodAdd', 'OverallCond', 'TotRmsAbvGrd']


In the cell below, replace `None` to build a `ColumnTransformer` that keeps only the columns in `columns_needing_preprocessing` and `passthrough_columns`. We'll use an empty `FunctionTransformer` as a placeholder transformer for each. (In other words, there is no actual transformation happening, we are only using `ColumnTransformer` to select columns for now.)


```python

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

relevant_cols_transformer = ColumnTransformer(transformers=[
    # Some columns will be used for preprocessing/feature engineering
    ("preprocess", FunctionTransformer(), columns_needing_preprocessing),
    # Some columns just pass through
    ("passthrough", FunctionTransformer(), passthrough_columns)
], remainder="drop")
```

Now, run this code to see if your `ColumnTransformer` was set up correctly:


```python

from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ("relevant_cols", relevant_cols_transformer)
])

pipe.fit_transform(X_train)

# Transform X_train and create dataframe for readability
X_train_transformed = pipe.fit_transform(X_train)
pd.DataFrame(
    X_train_transformed,
    columns=columns_needing_preprocessing + passthrough_columns
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FireplaceQu</th>
      <th>LotFrontage</th>
      <th>Street</th>
      <th>GrLivArea</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>OverallQual</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>FullBath</th>
      <th>Fireplaces</th>
      <th>YearRemodAdd</th>
      <th>OverallCond</th>
      <th>TotRmsAbvGrd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gd</td>
      <td>43</td>
      <td>Pave</td>
      <td>1504</td>
      <td>3182</td>
      <td>2005</td>
      <td>7</td>
      <td>2</td>
      <td>2008</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fa</td>
      <td>78</td>
      <td>Pave</td>
      <td>1309</td>
      <td>10140</td>
      <td>1974</td>
      <td>6</td>
      <td>3</td>
      <td>2006</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1999</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>60</td>
      <td>Pave</td>
      <td>1258</td>
      <td>9060</td>
      <td>1939</td>
      <td>6</td>
      <td>2</td>
      <td>2009</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1950</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>NaN</td>
      <td>Pave</td>
      <td>1422</td>
      <td>12342</td>
      <td>1960</td>
      <td>5</td>
      <td>3</td>
      <td>2007</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1978</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>75</td>
      <td>Pave</td>
      <td>1442</td>
      <td>9750</td>
      <td>1958</td>
      <td>6</td>
      <td>4</td>
      <td>2007</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1958</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Gd</td>
      <td>78</td>
      <td>Pave</td>
      <td>1314</td>
      <td>9317</td>
      <td>2006</td>
      <td>6</td>
      <td>3</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>TA</td>
      <td>65</td>
      <td>Pave</td>
      <td>1981</td>
      <td>7804</td>
      <td>1928</td>
      <td>4</td>
      <td>4</td>
      <td>2009</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1950</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>NaN</td>
      <td>60</td>
      <td>Pave</td>
      <td>864</td>
      <td>8172</td>
      <td>1955</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1990</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Gd</td>
      <td>55</td>
      <td>Pave</td>
      <td>1426</td>
      <td>7642</td>
      <td>1918</td>
      <td>7</td>
      <td>3</td>
      <td>2007</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1998</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>TA</td>
      <td>53</td>
      <td>Pave</td>
      <td>1555</td>
      <td>3684</td>
      <td>2007</td>
      <td>7</td>
      <td>2</td>
      <td>2009</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2007</td>
      <td>5</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



(If you get the error message `ValueError: No valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask is allowed`, make sure you actually replaced all of the `None` values above.)

If you're getting stuck here, look at the solution branch in order to move forward.

Great! Now we have only the 15 relevant columns selected. They are in a different order, but the overall effect is the same as the `drop_irrelevant_columns` function above. The pipeline structure looks like this:


```python
from sklearn import set_config
set_config(display='diagram')
pipe
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="54876c15-248f-47ab-b389-72744fca87b0" type="checkbox" ><label class="sk-toggleable__label" for="54876c15-248f-47ab-b389-72744fca87b0">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('relevant_cols',
                 ColumnTransformer(transformers=[('preprocess',
                                                  FunctionTransformer(),
                                                  ['FireplaceQu', 'LotFrontage',
                                                   'Street', 'GrLivArea',
                                                   'LotArea', 'YearBuilt',
                                                   'OverallQual']),
                                                 ('passthrough',
                                                  FunctionTransformer(),
                                                  ['BedroomAbvGr', 'YrSold',
                                                   'MoSold', 'FullBath',
                                                   'Fireplaces', 'YearRemodAdd',
                                                   'OverallCond',
                                                   'TotRmsAbvGrd'])]))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ba5bf042-71ac-4991-8a8a-e5a7aa072d47" type="checkbox" ><label class="sk-toggleable__label" for="ba5bf042-71ac-4991-8a8a-e5a7aa072d47">relevant_cols: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('preprocess', FunctionTransformer(),
                                 ['FireplaceQu', 'LotFrontage', 'Street',
                                  'GrLivArea', 'LotArea', 'YearBuilt',
                                  'OverallQual']),
                                ('passthrough', FunctionTransformer(),
                                 ['BedroomAbvGr', 'YrSold', 'MoSold',
                                  'FullBath', 'Fireplaces', 'YearRemodAdd',
                                  'OverallCond', 'TotRmsAbvGrd'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="620ceb36-c0e4-4f16-9910-4fa4bbf278a4" type="checkbox" ><label class="sk-toggleable__label" for="620ceb36-c0e4-4f16-9910-4fa4bbf278a4">preprocess</label><div class="sk-toggleable__content"><pre>['FireplaceQu', 'LotFrontage', 'Street', 'GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="79e6e112-bcd5-49ee-9fe0-66c7e2f446d5" type="checkbox" ><label class="sk-toggleable__label" for="79e6e112-bcd5-49ee-9fe0-66c7e2f446d5">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="da2c96c5-c777-4d2b-8939-62618ae0308e" type="checkbox" ><label class="sk-toggleable__label" for="da2c96c5-c777-4d2b-8939-62618ae0308e">passthrough</label><div class="sk-toggleable__content"><pre>['BedroomAbvGr', 'YrSold', 'MoSold', 'FullBath', 'Fireplaces', 'YearRemodAdd', 'OverallCond', 'TotRmsAbvGrd']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="20901918-f798-4e9d-86a9-b68297a76986" type="checkbox" ><label class="sk-toggleable__label" for="20901918-f798-4e9d-86a9-b68297a76986">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer()</pre></div></div></div></div></div></div></div></div></div></div></div></div>



You can click on the various elements (e.g. "relevant_cols: ColumnTransformer") to see more details.

### 2. Handle Missing Values

Same as before, we actually have two parts of handling missing values:

* Imputing missing values for `FireplaceQu` and `LotFrontage`
* Adding a missing indicator column for `LotFrontage`


Let's start with imputing missing values.

#### Imputing `FireplaceQu`

The NaNs in `FireplaceQu` (fireplace quality) are not really "missing" data, they just reflect homes without fireplaces. Previously we simply used pandas to replace these values:

```python
X["FireplaceQu"] = X["FireplaceQu"].fillna("N/A")
```

In a pipeline, we want to use a `SimpleImputer` to achieve the same thing. One of the available "strategies" of a `SimpleImputer` is "constant", meaning we fill every NaN with the same value.

Let's nest this logic inside of a pipeline, because we know we'll also need to one-hot encode `FireplaceQu` eventually.

In the cell below, replace `None` to specify the list of columns that this transformer should apply to:


```python

# Pipeline for all FireplaceQu steps
fireplace_qu_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="N/A"))
])

# ColumnTransformer for columns requiring preprocessing only
preprocess_cols_transformer = ColumnTransformer(transformers=[
    ("fireplace_qu", fireplace_qu_pipe, ["FireplaceQu"])
], remainder="passthrough")
```

Now run this code to see if that `ColumnTransformer` is correct:


```python

relevant_cols_transformer = ColumnTransformer(transformers=[
    # Some columns will be used for preprocessing/feature engineering
    ("preprocess", preprocess_cols_transformer, columns_needing_preprocessing),
    # Some columns just pass through
    ("passthrough", FunctionTransformer(), passthrough_columns)
], remainder="drop")

pipe = Pipeline(steps=[
    ("relevant_cols", relevant_cols_transformer)
])

pipe.fit_transform(X_train)

# Transform X_train and create dataframe for readability
X_train_transformed = pipe.fit_transform(X_train)
pd.DataFrame(
    X_train_transformed,
    columns=columns_needing_preprocessing + passthrough_columns
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FireplaceQu</th>
      <th>LotFrontage</th>
      <th>Street</th>
      <th>GrLivArea</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>OverallQual</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>FullBath</th>
      <th>Fireplaces</th>
      <th>YearRemodAdd</th>
      <th>OverallCond</th>
      <th>TotRmsAbvGrd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gd</td>
      <td>43</td>
      <td>Pave</td>
      <td>1504</td>
      <td>3182</td>
      <td>2005</td>
      <td>7</td>
      <td>2</td>
      <td>2008</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fa</td>
      <td>78</td>
      <td>Pave</td>
      <td>1309</td>
      <td>10140</td>
      <td>1974</td>
      <td>6</td>
      <td>3</td>
      <td>2006</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1999</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>1258</td>
      <td>9060</td>
      <td>1939</td>
      <td>6</td>
      <td>2</td>
      <td>2009</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1950</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>NaN</td>
      <td>Pave</td>
      <td>1422</td>
      <td>12342</td>
      <td>1960</td>
      <td>5</td>
      <td>3</td>
      <td>2007</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1978</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N/A</td>
      <td>75</td>
      <td>Pave</td>
      <td>1442</td>
      <td>9750</td>
      <td>1958</td>
      <td>6</td>
      <td>4</td>
      <td>2007</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1958</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Gd</td>
      <td>78</td>
      <td>Pave</td>
      <td>1314</td>
      <td>9317</td>
      <td>2006</td>
      <td>6</td>
      <td>3</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>TA</td>
      <td>65</td>
      <td>Pave</td>
      <td>1981</td>
      <td>7804</td>
      <td>1928</td>
      <td>4</td>
      <td>4</td>
      <td>2009</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1950</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>864</td>
      <td>8172</td>
      <td>1955</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1990</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Gd</td>
      <td>55</td>
      <td>Pave</td>
      <td>1426</td>
      <td>7642</td>
      <td>1918</td>
      <td>7</td>
      <td>3</td>
      <td>2007</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1998</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>TA</td>
      <td>53</td>
      <td>Pave</td>
      <td>1555</td>
      <td>3684</td>
      <td>2007</td>
      <td>7</td>
      <td>2</td>
      <td>2009</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2007</td>
      <td>5</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



(If you get `ValueError: 1D data passed to a transformer that expects 2D data. Try to specify the column selection as a list of one item instead of a scalar.`, make sure you specified a *list* of column names, not just the column name. It should be a list of length 1.)

Now we can see "N/A" instead of "NaN" in those `FireplaceQu` records. We can also look at the structure of the pipeline, which is more complex now:


```python
pipe
```




<style>div.sk-top-container {color: black;background-color: white;}div.sk-toggleable {background-color: white;}label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.2em 0.3em;box-sizing: border-box;text-align: center;}div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}div.sk-estimator {font-family: monospace;background-color: #f0f8ff;margin: 0.25em 0.25em;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;}div.sk-estimator:hover {background-color: #d4ebff;}div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;}div.sk-item {z-index: 1;}div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}div.sk-parallel-item:only-child::after {width: 0;}div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0.2em;box-sizing: border-box;padding-bottom: 0.1em;background-color: white;position: relative;}div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}div.sk-label-container {position: relative;z-index: 2;text-align: center;}div.sk-container {display: inline-block;position: relative;}</style><div class="sk-top-container"><div class="sk-container"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="440d2909-4886-442b-8f5f-43b0463bfcb4" type="checkbox" ><label class="sk-toggleable__label" for="440d2909-4886-442b-8f5f-43b0463bfcb4">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[('relevant_cols',
                 ColumnTransformer(transformers=[('preprocess',
                                                  ColumnTransformer(remainder='passthrough',
                                                                    transformers=[('fireplace_qu',
                                                                                   Pipeline(steps=[('impute',
                                                                                                    SimpleImputer(fill_value='N/A',
                                                                                                                  strategy='constant'))]),
                                                                                   ['FireplaceQu'])]),
                                                  ['FireplaceQu', 'LotFrontage',
                                                   'Street', 'GrLivArea',
                                                   'LotArea', 'YearBuilt',
                                                   'OverallQual']),
                                                 ('passthrough',
                                                  FunctionTransformer(),
                                                  ['BedroomAbvGr', 'YrSold',
                                                   'MoSold', 'FullBath',
                                                   'Fireplaces', 'YearRemodAdd',
                                                   'OverallCond',
                                                   'TotRmsAbvGrd'])]))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c522bbb9-7d9b-41f2-85ef-ad350ee54ade" type="checkbox" ><label class="sk-toggleable__label" for="c522bbb9-7d9b-41f2-85ef-ad350ee54ade">relevant_cols: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[('preprocess',
                                 ColumnTransformer(remainder='passthrough',
                                                   transformers=[('fireplace_qu',
                                                                  Pipeline(steps=[('impute',
                                                                                   SimpleImputer(fill_value='N/A',
                                                                                                 strategy='constant'))]),
                                                                  ['FireplaceQu'])]),
                                 ['FireplaceQu', 'LotFrontage', 'Street',
                                  'GrLivArea', 'LotArea', 'YearBuilt',
                                  'OverallQual']),
                                ('passthrough', FunctionTransformer(),
                                 ['BedroomAbvGr', 'YrSold', 'MoSold',
                                  'FullBath', 'Fireplaces', 'YearRemodAdd',
                                  'OverallCond', 'TotRmsAbvGrd'])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="97f305df-15fa-4879-968d-bf43310ba04c" type="checkbox" ><label class="sk-toggleable__label" for="97f305df-15fa-4879-968d-bf43310ba04c">preprocess</label><div class="sk-toggleable__content"><pre>['FireplaceQu', 'LotFrontage', 'Street', 'GrLivArea', 'LotArea', 'YearBuilt', 'OverallQual']</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="fa30a4a2-f1cb-4965-9948-5902c65bfde3" type="checkbox" ><label class="sk-toggleable__label" for="fa30a4a2-f1cb-4965-9948-5902c65bfde3">fireplace_qu</label><div class="sk-toggleable__content"><pre>['FireplaceQu']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="996edf20-0bee-498a-baed-e6c1e2fbfb94" type="checkbox" ><label class="sk-toggleable__label" for="996edf20-0bee-498a-baed-e6c1e2fbfb94">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(fill_value='N/A', strategy='constant')</pre></div></div></div></div></div></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f26b5ffb-05ef-4807-934f-1ccb2acdf355" type="checkbox" ><label class="sk-toggleable__label" for="f26b5ffb-05ef-4807-934f-1ccb2acdf355">passthrough</label><div class="sk-toggleable__content"><pre>['BedroomAbvGr', 'YrSold', 'MoSold', 'FullBath', 'Fireplaces', 'YearRemodAdd', 'OverallCond', 'TotRmsAbvGrd']</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2599237c-6d36-4b48-8d50-a76914846e03" type="checkbox" ><label class="sk-toggleable__label" for="2599237c-6d36-4b48-8d50-a76914846e03">FunctionTransformer</label><div class="sk-toggleable__content"><pre>FunctionTransformer()</pre></div></div></div></div></div></div></div></div></div></div></div></div>



#### Imputing `LotFrontage`

This is actually a bit simpler than the `FireplaceQu` imputation, since `LotFrontage` can be used for modeling as soon as it is imputed (rather than requiring additional conversion of categories to numbers). So, we don't need to create a pipeline for `LotFrontage`, we just need to add another transformer tuple to `preprocess_cols_transformer`.

This time we left all three parts of the tuple as `None`. Those values need to be:

* A string giving this step a name
* A `SimpleImputer` with `strategy="median"`
* A list of relevant columns (just `LotFrontage` here)


```python

preprocess_cols_transformer = ColumnTransformer(transformers=[
    ("fireplace_qu", fireplace_qu_pipe, ["FireplaceQu"]),
    (
        "impute_frontage",
        SimpleImputer(strategy="median"),
        ["LotFrontage"]
    )
], remainder="passthrough")
```

Now that we've updated the `preprocess_cols_transformer`, check that the NaNs are gone from `LotFrontage`:


```python

relevant_cols_transformer = ColumnTransformer(transformers=[
    # Some columns will be used for preprocessing/feature engineering
    ("preprocess", preprocess_cols_transformer, columns_needing_preprocessing),
    # Some columns just pass through
    ("passthrough", FunctionTransformer(), passthrough_columns)
], remainder="drop")

pipe = Pipeline(steps=[
    ("relevant_cols", relevant_cols_transformer)
])

pipe.fit_transform(X_train)

# Transform X_train and create dataframe for readability
X_train_transformed = pipe.fit_transform(X_train)
pd.DataFrame(
    X_train_transformed,
    columns=columns_needing_preprocessing + passthrough_columns
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FireplaceQu</th>
      <th>LotFrontage</th>
      <th>Street</th>
      <th>GrLivArea</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>OverallQual</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>FullBath</th>
      <th>Fireplaces</th>
      <th>YearRemodAdd</th>
      <th>OverallCond</th>
      <th>TotRmsAbvGrd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gd</td>
      <td>43</td>
      <td>Pave</td>
      <td>1504</td>
      <td>3182</td>
      <td>2005</td>
      <td>7</td>
      <td>2</td>
      <td>2008</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fa</td>
      <td>78</td>
      <td>Pave</td>
      <td>1309</td>
      <td>10140</td>
      <td>1974</td>
      <td>6</td>
      <td>3</td>
      <td>2006</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1999</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>1258</td>
      <td>9060</td>
      <td>1939</td>
      <td>6</td>
      <td>2</td>
      <td>2009</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1950</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>70</td>
      <td>Pave</td>
      <td>1422</td>
      <td>12342</td>
      <td>1960</td>
      <td>5</td>
      <td>3</td>
      <td>2007</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1978</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N/A</td>
      <td>75</td>
      <td>Pave</td>
      <td>1442</td>
      <td>9750</td>
      <td>1958</td>
      <td>6</td>
      <td>4</td>
      <td>2007</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1958</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Gd</td>
      <td>78</td>
      <td>Pave</td>
      <td>1314</td>
      <td>9317</td>
      <td>2006</td>
      <td>6</td>
      <td>3</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>TA</td>
      <td>65</td>
      <td>Pave</td>
      <td>1981</td>
      <td>7804</td>
      <td>1928</td>
      <td>4</td>
      <td>4</td>
      <td>2009</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1950</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>864</td>
      <td>8172</td>
      <td>1955</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1990</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Gd</td>
      <td>55</td>
      <td>Pave</td>
      <td>1426</td>
      <td>7642</td>
      <td>1918</td>
      <td>7</td>
      <td>3</td>
      <td>2007</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1998</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>TA</td>
      <td>53</td>
      <td>Pave</td>
      <td>1555</td>
      <td>3684</td>
      <td>2007</td>
      <td>7</td>
      <td>2</td>
      <td>2009</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2007</td>
      <td>5</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



#### Adding a Missing Indicator

If you recall from the previous lesson, a `FeatureUnion` is useful when you want to combine engineered features with original (but preprocessed) features.

In this case, we are treating a `MissingIndicator` as an engineered feature, which should appear as the last column in our `X` data regardless of whether there are actually any missing values in `LotFrontage`.

First, let's refactor our entire pipeline so far, so that it uses a `FeatureUnion`:


```python

from sklearn.pipeline import FeatureUnion

### Original features ###

# Preprocess fireplace quality
fireplace_qu_pipe = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="N/A"))
])

# ColumnTransformer for columns requiring preprocessing
preprocess_cols_transformer = ColumnTransformer(transformers=[
    ("fireplace_qu", fireplace_qu_pipe, ["FireplaceQu"]),
    ("impute_frontage", SimpleImputer(strategy="median"), ["LotFrontage"])
], remainder="passthrough")

# ColumnTransformer for all original features that we want to keep
relevant_cols_transformer = ColumnTransformer(transformers=[
    ("preprocess", preprocess_cols_transformer, columns_needing_preprocessing),
    ("passthrough", FunctionTransformer(), passthrough_columns)
], remainder="drop")

### Combine all features ###

# Feature union (currently only contains original features)
feature_union = FeatureUnion(transformer_list=[
    ("original_features", relevant_cols_transformer)
])

# Pipeline (currently only contains feature union)
pipe = Pipeline(steps=[
    ("all_features", feature_union)
])
pipe.fit_transform(X_train)

# Transform X_train and create dataframe for readability
X_train_transformed = pipe.fit_transform(X_train)
pd.DataFrame(
    X_train_transformed,
    columns=columns_needing_preprocessing + passthrough_columns
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FireplaceQu</th>
      <th>LotFrontage</th>
      <th>Street</th>
      <th>GrLivArea</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>OverallQual</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>FullBath</th>
      <th>Fireplaces</th>
      <th>YearRemodAdd</th>
      <th>OverallCond</th>
      <th>TotRmsAbvGrd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gd</td>
      <td>43</td>
      <td>Pave</td>
      <td>1504</td>
      <td>3182</td>
      <td>2005</td>
      <td>7</td>
      <td>2</td>
      <td>2008</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fa</td>
      <td>78</td>
      <td>Pave</td>
      <td>1309</td>
      <td>10140</td>
      <td>1974</td>
      <td>6</td>
      <td>3</td>
      <td>2006</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1999</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>1258</td>
      <td>9060</td>
      <td>1939</td>
      <td>6</td>
      <td>2</td>
      <td>2009</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1950</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>70</td>
      <td>Pave</td>
      <td>1422</td>
      <td>12342</td>
      <td>1960</td>
      <td>5</td>
      <td>3</td>
      <td>2007</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1978</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N/A</td>
      <td>75</td>
      <td>Pave</td>
      <td>1442</td>
      <td>9750</td>
      <td>1958</td>
      <td>6</td>
      <td>4</td>
      <td>2007</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1958</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Gd</td>
      <td>78</td>
      <td>Pave</td>
      <td>1314</td>
      <td>9317</td>
      <td>2006</td>
      <td>6</td>
      <td>3</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>TA</td>
      <td>65</td>
      <td>Pave</td>
      <td>1981</td>
      <td>7804</td>
      <td>1928</td>
      <td>4</td>
      <td>4</td>
      <td>2009</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1950</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>864</td>
      <td>8172</td>
      <td>1955</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1990</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Gd</td>
      <td>55</td>
      <td>Pave</td>
      <td>1426</td>
      <td>7642</td>
      <td>1918</td>
      <td>7</td>
      <td>3</td>
      <td>2007</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1998</td>
      <td>8</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>TA</td>
      <td>53</td>
      <td>Pave</td>
      <td>1555</td>
      <td>3684</td>
      <td>2007</td>
      <td>7</td>
      <td>2</td>
      <td>2009</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2007</td>
      <td>5</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 15 columns</p>
</div>



Now we can add another item to the `FeatureUnion`!

Specifically, we want a `MissingIndicator` that applies only to `LotFrontage`. So that means we need a new `ColumnTransformer` with a `MissingIndicator` inside of it.

In the cell below, replace `None` to complete the new `ColumnTransformer`.

Now run the following code to add this `ColumnTransformer` to the `FeatureUnion` and fit a new pipeline. If you scroll all the way to the right, you should see a new column, `LotFrontage_Missing`!


```python

# Feature union (currently only contains original features)
feature_union = FeatureUnion(transformer_list=[
    ("original_features", relevant_cols_transformer),
    ("engineered_features", feature_eng)
])

# Pipeline (currently only contains feature union)
pipe = Pipeline(steps=[
    ("all_features", feature_union)
])
pipe.fit_transform(X_train)

# Transform X_train and create dataframe for readability
X_train_transformed = pipe.fit_transform(X_train)
pd.DataFrame(
    X_train_transformed,
    columns=columns_needing_preprocessing + passthrough_columns + ["LotFrontage_Missing"]
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FireplaceQu</th>
      <th>LotFrontage</th>
      <th>Street</th>
      <th>GrLivArea</th>
      <th>LotArea</th>
      <th>YearBuilt</th>
      <th>OverallQual</th>
      <th>BedroomAbvGr</th>
      <th>YrSold</th>
      <th>MoSold</th>
      <th>FullBath</th>
      <th>Fireplaces</th>
      <th>YearRemodAdd</th>
      <th>OverallCond</th>
      <th>TotRmsAbvGrd</th>
      <th>LotFrontage_Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gd</td>
      <td>43</td>
      <td>Pave</td>
      <td>1504</td>
      <td>3182</td>
      <td>2005</td>
      <td>7</td>
      <td>2</td>
      <td>2008</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fa</td>
      <td>78</td>
      <td>Pave</td>
      <td>1309</td>
      <td>10140</td>
      <td>1974</td>
      <td>6</td>
      <td>3</td>
      <td>2006</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1999</td>
      <td>6</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>1258</td>
      <td>9060</td>
      <td>1939</td>
      <td>6</td>
      <td>2</td>
      <td>2009</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>1950</td>
      <td>5</td>
      <td>6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TA</td>
      <td>70</td>
      <td>Pave</td>
      <td>1422</td>
      <td>12342</td>
      <td>1960</td>
      <td>5</td>
      <td>3</td>
      <td>2007</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1978</td>
      <td>5</td>
      <td>6</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>N/A</td>
      <td>75</td>
      <td>Pave</td>
      <td>1442</td>
      <td>9750</td>
      <td>1958</td>
      <td>6</td>
      <td>4</td>
      <td>2007</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1958</td>
      <td>6</td>
      <td>7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>Gd</td>
      <td>78</td>
      <td>Pave</td>
      <td>1314</td>
      <td>9317</td>
      <td>2006</td>
      <td>6</td>
      <td>3</td>
      <td>2007</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2006</td>
      <td>5</td>
      <td>6</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>TA</td>
      <td>65</td>
      <td>Pave</td>
      <td>1981</td>
      <td>7804</td>
      <td>1928</td>
      <td>4</td>
      <td>4</td>
      <td>2009</td>
      <td>12</td>
      <td>2</td>
      <td>2</td>
      <td>1950</td>
      <td>3</td>
      <td>7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>N/A</td>
      <td>60</td>
      <td>Pave</td>
      <td>864</td>
      <td>8172</td>
      <td>1955</td>
      <td>5</td>
      <td>2</td>
      <td>2006</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1990</td>
      <td>7</td>
      <td>5</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>Gd</td>
      <td>55</td>
      <td>Pave</td>
      <td>1426</td>
      <td>7642</td>
      <td>1918</td>
      <td>7</td>
      <td>3</td>
      <td>2007</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>1998</td>
      <td>8</td>
      <td>7</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>TA</td>
      <td>53</td>
      <td>Pave</td>
      <td>1555</td>
      <td>3684</td>
      <td>2007</td>
      <td>7</td>
      <td>2</td>
      <td>2009</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>2007</td>
      <td>5</td>
      <td>7</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1095 rows × 16 columns</p>
</div>



Now we should have a dataframe with 16 columns: our original 15 relevant columns (in various states of preprocessing completion) plus a new engineered column.
