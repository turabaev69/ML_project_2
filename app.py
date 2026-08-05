from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

eng_yaxshi_model = joblib.load('sugurta_modeli.pkl')
scaler = joblib.load('scaler.pkl')


ustunlar_tartibi = [
    'age', 'bmi', 'children', 'is_female', 'is_smoker',
    'region_northwest', 'region_southeast', 'region_southwest',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]



def sugurta_narxini_bashorat_qil(yosh, bmi, farzandlar_soni, ayolmi,
                                  chekadimi, hudud, model, scaler, ustunlar):
    if bmi < 18.5:
        bmi_kategoriya = 'Underweight'
    elif bmi < 25:
        bmi_kategoriya = 'Normal'
    elif bmi < 30:
        bmi_kategoriya = 'Overweight'
    else:
        bmi_kategoriya = 'Obese'

    yangi_mijoz = pd.DataFrame([{
        'age': yosh,
        'bmi': bmi,
        'children': farzandlar_soni,
        'is_female': int(ayolmi),
        'is_smoker': int(chekadimi),
        'region_northwest': 1 if hudud == 'northwest' else 0,
        'region_southeast': 1 if hudud == 'southeast' else 0,
        'region_southwest': 1 if hudud == 'southwest' else 0,
        'bmi_category_Normal': 1 if bmi_kategoriya == 'Normal' else 0,
        'bmi_category_Overweight': 1 if bmi_kategoriya == 'Overweight' else 0,
        'bmi_category_Obese': 1 if bmi_kategoriya == 'Obese' else 0,
    }])

    yangi_mijoz[['age', 'bmi', 'children']] = scaler.transform(
        yangi_mijoz[['age', 'bmi', 'children']]
    )
    yangi_mijoz = yangi_mijoz[ustunlar]

    natija = model.predict(yangi_mijoz)
    return natija[0]


@app.route('/')
def hello():
    return "hello world!"

@app.route('/predict', methods=["GET", "POST"])
def predict():
    natija = None

    if request.method == 'POST':
        yosh = request.form.get('yosh', type=float)
        bmi = request.form.get('bmi', type=float)
        farzandlar_soni = request.form.get('children', type=int)
        ayolmi = request.form.get('gender') == '1'
        chekadimi = request.form.get('chekadimi') == '1'
        hudud = request.form.get('hudud')

        natija = sugurta_narxini_bashorat_qil(
            yosh=yosh,
            bmi=bmi,
            farzandlar_soni=farzandlar_soni,
            ayolmi=ayolmi,
            chekadimi=chekadimi,
            hudud=hudud,
            model=eng_yaxshi_model,
            scaler=scaler,
            ustunlar=ustunlar_tartibi
        )
        natija = round(float(natija), 2)

    return render_template('index.html', natija=natija)




# @app.route('/predict', methods=['POST'])
# def predict():
#      if request.method == 'GET':
#             return "Use a POST request with JSON data."

    
#      data = request.get_json() 
       

#      natija = sugurta_narxini_bashorat_qil(
#             yosh=data['age'],
#             bmi=data['bmi'],
#             farzandlar_soni=data['children'],
#             ayolmi=data['is_female'],
#             chekadimi=data['is_smoker'],
#             hudud=data['region'],
#             model=eng_yaxshi_model,
#             scaler=scaler,
#             ustunlar=ustunlar_tartibi
#         )

#      return jsonify({"sugurta_narxi": float(natija)})

   



if __name__ == "__main__":
    app.run(debug=True)