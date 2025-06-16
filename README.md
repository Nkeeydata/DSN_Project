# DSN_Project

## This is a project/hackanthon is to predict the outcome of profits in different stores at various location owns by one man.

### PROBLEM DEFINITION : A business man with retail shops across the locations, all in Nigeria. He wants to know how he can make more profits without selling same price across his variou stores.

```python


merged_df['Store_Size'].fillna(merged_df['Store_Size'].mode()[0], inplace=True)
merged_df['Item_Weight'].fillna(merged_df['Item_Weight'].mean(), inplace=True)
```

```python

cat_vars = merged_df.select_dtypes(exclude=np.number)
num_vars = merged_df.select_dtypes(include=np.number)
```

```python
one_hot = OneHotEncoder(sparse_output = False)
encoder = one_hot.fit_transform(cat_vars)
encoder
cat_col = one_hot.get_feature_names_out()
df_cat = pd.DataFrame(encoder, columns = cat_col) # converting our encoded DataFrame into pandas DataFrame
df_cat
```

```python
std_scale = StandardScaler() #assigning the object to a name

scaled = std_scale.fit_transform(num_vars) #performing transformation on the variable

df_num = pd.DataFrame(scaled, columns=num_vars.columns) #converting the scaled object to a pandas dataframe

df_num # displaying the new dataframe
```
```
df_new = pd.concat([df_cat, df_num], axis = 1)
```

```
train_size = len(trn_data)
test_size = len(tst_data)

df_train = df_new.iloc[:train_size, :]  # Extract first 'train_size' rows
df_test = df_new.iloc[train_size:, :]   # Extract remaining 'test_size' rows
```

```
print(df_train.shape)  # Should match original train data shape
print(df_test.shape)  # Should match original test data shape

```

```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
```

```python

model = LinearRegression() 

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

```

```
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")
```

