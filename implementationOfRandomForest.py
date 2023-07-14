import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#read data
df = pd.read_csv("melb_data.csv")

#prediction target
y = df.Price

#training data, validation data
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# build model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(train_X, train_y)
pre = rf_model.predict(val_X)

#split data
temp_df = df.loc[:100, 'Address': 'Bathroom']
temp_df.to_csv('test.csv')

#find best tree size
def get_mae(mln, train_X, val_X, train_y, val_y):
    model = RandomForestClassifier(max_leaf_nodes=mln, random_state=0)
    model.fit(train_X, train_y)
    predict = model.predict(val_X)
    return mean_absolute_error(val_y, predict)

for i in [5, 50, 500, 5000]:
    my_mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d\t\t Mean absolute error: %d" %(i, my_mae))

result = {i: get_mae(i, train_X, val_X, train_y, val_y) for i in [5, 50, 500, 5000]}
best_size = min(result, key=result.get)
print("The best tree size: %d\t\t with MAE: %d" %(best_size, get_mae(best_size, train_X, val_X, train_y, val_y)))