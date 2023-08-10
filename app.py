from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

pickled_model = pickle.load(open('customer_churn_predictor.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    credit_score = float(request.form['credit_score'])
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    balance = float(request.form['balance'])
    num_of_products = int(request.form['num_of_products'])
    has_credit_card = int(request.form['has_credit_card'])
    is_active_member = int(request.form['is_active_member'])
    estimated_salary = float(request.form['estimated_salary'])
    is_female = int(request.form['is_female'])
    is_spain = int(request.form['is_spain'])
    is_germany = int(request.form['is_germany'])

    #UserInput to list
    input_data = [[credit_score, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary, is_female, is_spain, is_germany]]

    output = pickled_model.predict(input_data)
    if output[0] == 1:
        prediction = "The customer is predicted to churn."
    else:
        prediction = "The customer is predicted not to churn."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)