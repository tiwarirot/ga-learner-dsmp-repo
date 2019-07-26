# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

#Loading the data
data=pd.read_csv(path)

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')

plt.show()


#Subsetting the dataframe based on `Rating` column
data=data[data['Rating']<=5]

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')   

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
missing_data = pd.concat([total_null,percent_null], keys=['Total', 'Percent'], axis=1)
print(missing_data)
data = data.dropna()
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1,percent_null_1], keys=['Total', 'Percent'], axis=1)
print(missing_data_1)
# code ends here


# --------------

#Code starts here
sns.catplot(x='Category', y='Rating', data=data, kind='box', height=10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
plt.show()
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].str.replace(',', '')
data['Installs'] = data['Installs'].astype('Int64')
le = LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
sns.regplot(x='Installs', y='Rating', data=data)
plt.title('Rating vs Installs [RegPlot]')
plt.show()
#Code ends here



# --------------
#Code starts here
print(data['Price'].value_counts())

data['Price'] = data['Price'].str.replace('$', '')
data['Price'] = data['Price'].astype('float')
sns.regplot(x='Price', y='Rating',data=data)
plt.title('Rating vs Price [RegPlot]')
plt.show()
#Code ends here


# --------------

#Code starts here

print(data['Genres'].unique())
data['Genres'] = data['Genres'].str.split(';').str[0].tolist()
gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()
gr_mean.describe()
gr_mean = gr_mean.sort_values(by = 'Rating', ascending=True)
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])

#Code ends here


# --------------

#Code starts here

#Converting the column into datetime format
data['Last Updated'] = pd.to_datetime(data['Last Updated'])

#Creating new column having `Last Updated` in days
data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated'] ).dt.days 

#Setting the size of the figure
plt.figure(figsize = (10,10))

#Plotting a regression plot between `Rating` and `Last Updated Days`
sns.regplot(x="Last Updated Days", y="Rating", color = 'lightpink',data=data )

#Setting the title of the plot
plt.title('Rating vs Last Updated [RegPlot]',size = 20)

#Code ends here


