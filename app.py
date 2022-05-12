#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle


# In[ ]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('students.html')


@app.route('/students.html', methods=['POST', 'GET'])
def students():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        pred = "have a good performance"
    elif prediction == 1:
        pred = " need improvement"
    output = pred
    return render_template('students.html', prediction_text='You {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




