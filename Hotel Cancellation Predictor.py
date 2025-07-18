import  warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('C:\\Users\\sara\\Downloads\\hotels.csv')
pd.set_option("display.width",None)
print(df.head(25))

print("-------------------------------")
print("========== Basic Function:")
print("number of rows and columns:")
print(df.shape)
print("The name of columns:")
print(df.columns)
print("Information about data:")
print(df.info())
print("Statistical operations:")
print(df.describe().round())
print("Data types in Hotel data:")
print(df.dtypes)
print("Display the index range:")
print(df.index)
print("number of frequency rows:")
print(df.duplicated().sum())
print(df.isnull().sum())
print("-------------------------------")
print("========== Cleaning Data:")
missing_values = df.isnull().mean() * 100
print('The Percentage of missing values in data : \n',missing_values)
print("Missing Values Before Cleaning :")
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title("The Dataset Before Cleaning")
plt.show()
print("The missing value in children column = 0.003 so we use fillna = 0 ")
df['children'] = df['children'].fillna(0)
print("The missing value in country column = 0.408 so we use fillna = mode ")
df['country'] = df['country'].fillna(df['country'].mode()[0])
print("The missing value in agent column = 13.68 so we use fillna = mean ")
df['agent'] = df['agent'].fillna(df['agent'].mean())
print("The children,country,and agent columns contain miss values?")
print(df[['children','country','agent']].isnull().sum())
print("The missing value in company column = 94.306 so we use drop ")
df = df.drop(columns='company',axis=1)
print("The company column contain more miss value so we remove ")
print("Missing Values after Cleaning")
#print(df)
print(df.isnull().sum())
sns.heatmap(df.isnull())
plt.title('The Dataset After Cleaning')
plt.show()
#---------------------------------------
print("-------------------------------")
print("========== Exploration Data Analysis")
print("convert reservation_status_data column to datetime")
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
print(df.dtypes)
print("Statistical operation")
print(df.describe(include='object').round())
print("what are the unique values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].unique())
    print('-'*75)
print("what are the maximum values in data(object or text)")
#---------------------------------------
for col in df.describe(include='object').round():
    print(col)
    print(df[col].str.len().max())
    print('-'*75)
#---------------------------------------
df['adr'].plot(kind='box')
plt.grid()
plt.title('The Deviant values in adr column')
plt.show()
df = df[df['adr'] < 5000]
print(df.head(10))
#---------------------------------------
print("The percentage of people who canceled Reservation :")
canceled_perc = df['is_canceled'].value_counts(normalize=True)
print(canceled_perc)
print("The people who canceled reservation in City Hotel")
city_hotel = df[df['hotel'] == 'City Hotel']
print(city_hotel['is_canceled'].value_counts(normalize=True))
print("The people who canceled reservation in Resort Hotel")
resort_hotel = df[df['hotel'] == 'Resort Hotel']
print(resort_hotel['is_canceled'].value_counts(normalize=True))
print("Does the price have any impact on cancellations on Resort Hotel ?")
resort_hotel = resort_hotel.groupby("reservation_status_date")['adr'].mean()
print(resort_hotel)
print("Does the price have any impact on cancellations on City Hotel ?")
city_hotel = city_hotel.groupby("reservation_status_date")['adr'].mean()
print(city_hotel)
#---------------------------------------
print("The Top 10 Country Who Canceled Reservation")
canceled_data = df[df['is_canceled'] == 1]
top_10_country = canceled_data['country'].value_counts()[:10]
print(top_10_country)
#---------------------------------------
print("The Percentage of people who reserved on Hotels")
reserve = df['market_segment'].value_counts(normalize=True)
print(reserve)
#---------------------------------------
print("The Percentage of people who canceled reserve")
print(canceled_data['market_segment'].value_counts(normalize=True))
#---------------------------------------
print("Average price when canceling a reservation")
canceled_df_adr = canceled_data.groupby('reservation_status_date')['adr'].mean()
print(canceled_df_adr.reset_index)
print("Average price During reservation")
not_canceled = df[df['is_canceled'] == 0]
not_canceled_df_adr = canceled_data.groupby('reservation_status_date')['adr'].mean()
print(not_canceled_df_adr.reset_index)
print("-------------------------------")
print("========== Visualization Data")
plt.figure(figsize=(8,8))
plt.title("Reservation status count")
plt.bar(['Not Canceled','Canceled'],df['is_canceled'].value_counts(),edgecolor='black',width=0.7)
plt.show()
#---------------------------------------
df['is_canceled'] = df['is_canceled'].astype(str)
plt.figure(figsize=(8,6))
sns.countplot(x='hotel',hue='is_canceled',data=df,palette='Pastel1')
plt.title("Cancelation rates in City hotel and Resort hotel")
plt.show()
print("Most bookings were in city hotel")
print("Cancelation in Resort hotel is less than compared to city hotel")
#---------------------------------------
df['month'] = df['reservation_status_date'].dt.month
plt.figure(figsize=(15,7),facecolor='#C38154')
sns.countplot(x='month',hue='is_canceled',data=df,palette='Pastel1')
plt.title('Reservation status per month',fontweight='bold',size=20)
plt.legend(['not canceled','canceled'])
plt.grid()
plt.show()
#---------------------------------------
plt.figure(figsize=(15,8))
plt.title('ADR per month',fontsize=30)
data = df[df['is_canceled']=='1'].groupby('month')['adr'].sum().reset_index()
sns.barplot(x='month',y='adr',data=data)
plt.legend(['Canceled'])
plt.show()
print("The higher the price,The more cancellations there are.")
#---------------------------------------
canceled_data = df[df['is_canceled'] == '1']
top_10_country = canceled_data['country'].value_counts()[:10]
plt.figure(figsize=(8,8))
plt.title("Top 10 countries with reservation canceled",color='black')
plt.pie(top_10_country,labels=top_10_country.index,autopct='%1.1f%%')
plt.show()
#---------------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(x='arrival_date_month',y='adr',data=df,hue='hotel')
plt.grid()
plt.title("months are the highest prices ")
plt.show()

#-----------------------------------
"""
['hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
       'arrival_date_month', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'stays_in_weekend_nights',
       'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
       'country', 'market_segment', 'distribution_channel',
       'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'reserved_room_type',
       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
       'company', 'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status', 'reservation_status_date', 'name', 'email',
       'phone-number', 'credit_card']
"""
print("----------------------- Machine Learning ------------------")
features = ['lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','stays_in_week_nights',
            'stays_in_weekend_nights','adults', 'children', 'babies','previous_cancellations','previous_bookings_not_canceled',
            'total_of_special_requests']
print("=========>>> Bulding Model :")
X = df[features]
y = df['is_canceled']
# Deal With Missing Values if found and replace it by mean
X = pd.get_dummies(X,columns=['arrival_date_month'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.35,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
scaler =StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("=========>>> Model Training and Prediction >> Random Forest Classifier:")
model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')
print("model train:")
print(model.fit(X_train,y_train))
y_pred = model.predict(X_test)
print("y_predict:\n",y_pred)
print("y_test:\n",y_test.values)
print("model evaluation:")
print("confusion_matrix:\n",confusion_matrix(y_test,y_pred))
print("classification_report",classification_report(y_test,y_pred))
print("Mean absolute error:",mean_absolute_error(y_test,y_pred))
print("model.score:",model.score(X,y))
print("R2 Score:",r2_score(y_test,y_pred))
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
#--------------------------
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature':feature_names,'Importance':importances})
print(feature_importance_df.sort_values(by='Importance',ascending=False))
#--------------------------
plt.figure(figsize=(10,6))
plt.bar(feature_importance_df['Feature'],feature_importance_df['Importance'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.show()