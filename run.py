import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA

'''Importing the csv file using pandas read function'''

data = pd.read_csv("SensorData_question1.csv")

'''making copies of the columns'''

data["Original Input3"] = data["Input3"]
data["Original Input12"] = data["Input12"]

'''normalizing using z-score '''
'''I used the below formula to calculate the zscore for the column Input3'''

data[['Input3']] = (data[['Input3']] - data[['Input3']].mean()) / data[['Input3']].std()

'''normalizing in a range'''
'''As mentioned at the end of the practical2 file, I have used the minmax_scale from the Sklearn 
   Library to normalise everything within the range [0.0, 1.0]'''

data['Input12'] = minmax_scale(data['Input12'])

''' Average of all columns '''

data['Average Input'] = data.iloc[:, 0:12].mean(axis=1)

'''Saving to a csv file'''
'''I have removed the index column as it gives me an error while running the test file'''
'''I have added in a float argument so as to prevent any rounding errors'''

data.to_csv('./output/question1_out.csv', index=False, float_format= '%g')
#print(data)

''' Question 2'''
Data = pd.read_csv('DNAData_question2.csv')
Data_Copy = Data
pca = PCA(n_components=0.95)

Data = pca.fit_transform(Data)

''' Transforming the data back to a dataframe from an array'''

Data = pd.DataFrame(Data)
#print(Data)

'''Discretizing the newly created dataset into 10 columns of equal width'''
'''Using the pandas cut function to divide the data into columns'''

newBinsSecond = Data.apply(lambda colmns : pd.cut(colmns,10))
#print(newBins)

newList = list()

'''Storing the names of the columns in a list'''

for colmns in newBinsSecond: 
	newList.append("pca" + str(colmns)+ "_width")

'''Adding in the column name list into the data set'''

newBinsSecond.columns = newList


# question 2 part 3

'''Using the pandas qcut method to divide the data by frequency '''

newBinsThird = Data.apply(lambda colmns : pd.qcut(colmns,10))

'''A list for storing the column names'''
newListThird = list()

'''Storing the column names in the list '''
for colmns in newBinsThird:
	newListThird.append("pca" + str(colmns) + "_freq")

'''Adding the column names to the data set'''
newBinsThird.columns = newListThird

''' Addding the original dataset and the set of columns to the new dataset '''

Data = pd.concat([Data_Copy, newBinsSecond, newBinsThird], axis = 1)

'''Exporting to a csv file'''

Data.to_csv("./output/question2_out.csv", index = False)