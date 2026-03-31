from flask import Flask, render_template,request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('cv.pkl','rb'))

ps = PorterStemmer()

custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                    'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

stop_words = set(stopwords.words('english'))-custom_stopwords

def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return " ".join(review)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    user_input=request.form['review']
    processed = preprocess(user_input)

    vectorized = vectorizer.transform([processed]).toarray()
    predict = model.predict(vectorized)[0]
    result = "Positive 😊" if predict==1 else "negative😞"
    return render_template('index.html',prediction_text = result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)