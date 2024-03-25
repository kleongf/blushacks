import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle





df = pd.read_csv('Emotion_classify_Data.csv')

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Comment'])

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(text):
    options = ["Angry", "Happy", "Scared"]
    txt = vectorizer.transform([text])
    prediction = model.predict(txt)
    return options[prediction[0]]

print(predict("i am not afraid of the dark i am afraid of whats in the dark that i cannot see"))

'''
X = vectorizer.fit_transform(df['Comment'])

def transform_text(text):
    return vectorizer.transform(text)

emotion_mapping = {'anger': 0, 'joy': 1, 'fear': 2}
df['emotion'] = df['Emotion'].map(emotion_mapping)
X_train, X_test, y_train, y_test = train_test_split(X, df['emotion'], test_size=0.2, random_state=42, shuffle = False)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
#with open('model.pkl','wb') as f:
    #pickle.dump(nb,f)
print(classification_report(y_test, y_pred_nb))
print(nb.predict(transform_text(["spiders really creep me out"])))
'''






'''
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
'''
