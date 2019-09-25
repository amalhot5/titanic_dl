#%% [markdown]
# This is an attempt to use a fast_ai transfer learning model on the Titanic tutorial dataset as a test of the its handling of tabular data

#%%
from fastai.tabular import *
import os

#%%
path = os.getcwd()
df = pd.read_csv('train.csv')
df.head()

#%%
valid_idx = range(len(df)-180, len(df))
procs = [FillMissing, Categorify, Normalize]
#%%
df_train = df.drop(columns=['Name',  'PassengerId'])
print(df_train.head())
cat_names = ['Sex', 'Ticket', 'Cabin', 'Embarked','Pclass', 'SibSp', 'Parch']
dtypes = {cat: 'category' for cat in cat_names}
cont_names = ['Age', 'Fare']
dtypes.update({cont:'float32' for cont in cont_names})
df_train = df_train.astype(dtypes)
data = TabularDataBunch.from_df(path, df_train, 'Survived', valid_idx=valid_idx, cat_names=cat_names)
print(data.train_ds.cont_names)

#%%
(cat_x, cont_x), y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))

#%%
# defining a model
learn = tabular_learner(data, layers=[200,100], emb_szs={'Sex': 2}, metrics=accuracy)
learn.fit_one_cycle(1,1e-2)

#%%
