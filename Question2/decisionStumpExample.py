#https://medium.com/geekculture/decision-stump-b8e93c1f54d7
#https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
X = iris.data[:, :2] 
y = iris.target


iris  = pd.DataFrame(X)
iris['y'] = y

condition = (iris['y'] == 0) | (iris['y'] == 1)

iris = iris[condition]

iris.head(10)
print(iris)

iris['y_hat'] = 0
final_gini = 1
final_error = 1

stump_base_on_gini = 0
stump_base_on_error = 0

direction_base_on_gini = 0
direction_bast_on_error = 0

feature_base_on_gini = 0
feature_base_on_error = 0

def gini(df):
    print(df)
    gini = 0
    res_gini = 0
    for i in range(2):
        N = len(df[df['y_hat']==i])
        condition = (df['y'] == df['y_hat']) & (df['y_hat']==i)
        gini = 1 - (len(df[condition])/ N)**2
        
        res_gini += gini*N  # add gini by its weight
        
    return res_gini

def classification_error(df):
    return len(df[df['y'] != df['y_hat']])/len(df)

for i in range(2): # we only have two features
    for j in range(len(iris)):
        
        # direction 1
        a = iris[i].iloc[j]  # our a
        

        iris.loc[iris[0] <= a, 'y_hat'] = 0
        iris.loc[iris[0] > a, 'y_hat'] = 1
        
        temp_gini = gini(iris)
        temp_error = classification_error(iris)
        
        
        
        # pick one error funtion
        if(final_gini > temp_gini ):
            final_gini = temp_gini
            stump_base_on_gini = a
            direction_base_on_gini = 1
            feature_base_on_gini = i
            
        
        if(final_error > temp_error ):
            final_error = temp_error
            stump_base_on_error = a
            direction_bast_on_error = 1
            feature_base_on_error = i
        
        # direction -1
        iris.loc[iris[0] <= -1*a, 'y_hat'] = 0
        iris.loc[iris[0] > -1*a, 'y_hat'] = 1
        
        temp_gini = gini(iris)
        temp_error = classification_error(iris)
        
        
        # pick one error funtion
        if(final_gini > temp_gini ):
            final_gini = temp_gini
            stump_base_on_gini = a
            direction_base_on_gini = -1
            feature_base_on_gini = i
            
        
        if(final_error > temp_error ):
            final_error = temp_error
            stump_base_on_error = a
            direction_bast_on_error = -1
            feature_base_on_error = i
            

print(f"Stump base on Gini :{stump_base_on_gini}")
print(f"Feature :{feature_base_on_gini}")
print(f"Final Gini : {final_gini}")
print(f"Direction :{direction_base_on_gini}")


print(f"Stump base on error: {stump_base_on_error}")
print(f"Feature : {feature_base_on_error}")
print(f"Final error : {final_gini}")
print(f"Direction: {direction_bast_on_error}")
        