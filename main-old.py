import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import  cross_val_score

housing=pd.read_csv("housing.csv")
housing['income_cat']=pd.cut(housing["median_income"],bins=[0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5]);
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)
    strat_test_set=housing.loc[test_index].drop('income_cat',axis=1)
print(strat_test_set)
housing=strat_train_set.copy();
housing_labels=housing['median_house_value'].copy()
housing=housing.drop("median_house_value",axis=1)
num_attr=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attr=["ocean_proximity"]
num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
]

)
cat_pipeline=Pipeline(
    [
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ]
)
full_pipeline=ColumnTransformer(
    [
        ("num",num_pipeline,num_attr),
        ("cat",cat_pipeline,cat_attr)
    ]
)
housing_prepare=full_pipeline.fit_transform(housing)
print(full_pipeline.get_feature_names_out())
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepare, housing_labels)

lin_preds = lin_reg.predict(housing_prepare)
lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepare,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print(pd.Series(lin_rmses).describe())

# Decision Tree
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepare, housing_labels)

dec_preds = dec_reg.predict(housing_prepare)
dec_rmses = -cross_val_score(
    dec_reg,
    housing_prepare,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print(pd.Series(dec_rmses).describe())
# Random Forest
random_reg = RandomForestRegressor(random_state=42)
random_reg.fit(housing_prepare, housing_labels)

random_preds = random_reg.predict(housing_prepare)
random_rmses = -cross_val_score(
    random_reg,
    housing_prepare,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print(pd.Series(random_rmses).describe())
print(housing_prepared.columns)