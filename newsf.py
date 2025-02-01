from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

app = Flask(__name__)

# Sample dataset (you can replace this with a larger dataset)
data = {
    'text': [
        'The sky is blue',
        'The earth is flat',
        'Python is a programming language',
        'Aliens have landed on earth',
        'The sun rises in the east',
        'Vaccines cause autism',
        'Climate change is real',
        'The moon is made of cheese'
    ],
    'label': ['real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'fake']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create a model
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(df['text'], df['label'])

# Save the model
joblib.dump(model, 'news_model.pkl')

# Load the trained model
model = joblib.load('news_model.pkl')

# HTML templates
index_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 300px;
        }
        input {
            padding: 10px;
            margin-right: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            width: 70%;
        }
        button {
            padding: 10px 15px;
            border: none;
            background-color: #007BFF;
            color: white;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake News Detection</h1>
        <form action="/predict" method="post">
            <input type="text" name="headline" placeholder="Enter news headline" required>
            <button type="submit">Check</button>
        </form>
    </div>
</body>
</html>
"""

result_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 300px;
        }
        a {
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007BFF;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Result</h1>
        <p>Headline: <strong>{{ headline }}</strong></p>
        <p>This news is: <strong>{{ prediction }}</strong></p>
        <a href="/">Check another headline</a>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(index_template)

@app.route('/predict', methods=['POST'])
def predict():
    headline = request.form['headline']
    prediction = model.predict([headline])[0]
    return render_template_string(result_template, headline=headline, prediction=