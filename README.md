#  NeuroNosh: Restaurant Review Sentiment Predictor

NeuroNosh is a full-stack Machine Learning web application that classifies restaurant reviews as **Positive** or **Negative**. Built with Python, Flask, and Scikit-Learn, it uses Natural Language Processing (NLP) to understand customer feedback and provide real-time sentiment analysis.

##  Features
* **Real-time Prediction:** Enter any review and get instant feedback.
* **NLP Pipeline:** Custom preprocessing including regex cleaning, stopword removal (preserving negations), and Porter Stemming.
* **Machine Learning:** Utilizes a Random Forest Classifier for high-accuracy predictions (~80%).
* **Responsive UI:** A modern, glassmorphism-inspired interface built with Tailwind CSS.
* **Visual Insights:** Automated WordCloud generation for exploring common positive and negative terms.

##  Tech Stack
* **Frontend:** HTML5, Tailwind CSS
* **Backend:** Flask (Python)
* **ML/NLP:** Scikit-Learn, NLTK, Pandas, NumPy
* **Visualization:** Matplotlib, WordCloud
* **Deployment:** Gunicorn (Ready for Heroku/Render)

##  Model Performance
After testing multiple algorithms on the `Restaurant_Reviews.tsv` dataset, the results were:
* **Gaussian Naive Bayes:** 68.0%
* **Logistic Regression:** 79.5%
* **Random Forest Classifier:** 80.0% (Selected Model)

##  Project Structure
```text
├── app.py               # Flask application (Web Server)
├── main.py              # ML Pipeline: Training & EDA
├── model.pkl            # Serialized Random Forest Model
├── cv.pkl               # Serialized CountVectorizer
├── Restaurant_Reviews.tsv # Dataset for training
├── requirements.txt     # Python dependencies
├── Procfile             # Configuration for deployment (Render/Heroku)
├── templates/
│   └── index.html       # Frontend User Interface
├── static/
│   ├── output.css       # Minified Tailwind CSS styles
│   └── input.css        # Source Tailwind CSS styles
├── positive.jpg         # Generated WordCloud (Positive reviews)
├── negetive.jpg         # Generated WordCloud (Negative reviews)
├── .gitignore           # Files to exclude from Git (node_modules, etc.)
├── package.json         # Node.js metadata (for Tailwind/PostCSS)
└── README.md            # Project documentation