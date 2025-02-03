import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('D:\\Projects\\Datasets\\IMDB_Dataset.csv')  #reading csv file

print(data.head()) #printing first few lines of data

reviews = data["review"]    #taking out reviews (type series)
labels = data["sentiment"].map({'positive' : 1 , 'negative' : 0})  #taking out labels and assigning numbers for each output

vectorizer = TfidfVectorizer(stop_words='english' , max_features = 10000)

X = vectorizer.fit_transform(reviews)

Y = labels.astype(int)

X_train , X_test  , Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

model = LogisticRegression(max_iter = 1000)
model.fit(X_train,Y_train)

accuracy = accuracy_score(Y_test , model.predict(X_test))

print(f"Model trained successfully with Accuracy : {accuracy:.4f}")

joblib.dump(model,"p_model.pkl")
joblib.dump(vectorizer, "pvectorizer.pkl")

print("Model dumped successfully!")





