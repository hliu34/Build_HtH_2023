import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Call in our titanic csv
titanic_data = pd.read_csv('titanic.csv')

# Step 1: Create a heatmap showing the correlation between the following features: Survived, PassengerId, Sex, Age, and Fare
titanic_data['Sex'] = titanic_data['Sex'].map({'female': 0, 'male': 1})

# Select only the columns needed for the correlation heatmap
selected_columns = ['Survived', 'PassengerId', 'Sex', 'Age', 'Fare']
correlation_data = titanic_data[selected_columns].corr()

# Create the heatmap using seaborn
plt.figure(figsize=(5, 5))
corr_heatmap = sns.heatmap(correlation_data, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.show()

# Step 2: Create a bar graph that illustrates how many passengers fell into each of these age ranges: 0 - 16, 17 - 25, 26 -40, 41 - 59, 60 or older
age_bins = [0, 17, 26, 41, 60, titanic_data['Age'].max()]
age_labels = ['0-16', '17-25', '26-40', '41-59', '60+']
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=age_bins, labels=age_labels)
age_group_chart = titanic_data['AgeGroup'].value_counts().sort_index().plot(kind='bar')
age_group_chart.set_xlabel('Age Group')
age_group_chart.set_ylabel('Passenger Count')
plt.title('Passenger Count in Age Groups')
#plt.show()

# Step 3: Create a line graph showing the average survival percentage of each of the previous age groups

survival_percentage = titanic_data.groupby('AgeGroup')['Survived'].mean()
survival_percentage_chart = survival_percentage.plot(kind='line', marker='o')
survival_percentage_chart.set_xlabel('Age Group')
survival_percentage_chart.set_ylabel('Survival Percentage')
plt.title('Average Survival Percentage in Age Groups')
plt.show()


# Step 4: Create a pie chart that shows the percentage of survivors that were male and the percentage of survivors that were female
survivors = titanic_data['Survived'].value_counts()
plt.figure(figsize=(5, 5))
survivors.plot(kind='pie', autopct='%1.1f%%', startangle=90, labels=['Died', 'Survived'])
plt.title('Survivors Percentage by Gender')
plt.show()

# Step 5: Create a histogram that shows the distribution of passengers between the three embarking locations: C (Cherbourg), Q (Queenstown), S (Southampton)
#titanic_description = 
