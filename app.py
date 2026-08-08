from flask import Flask, render_template, request
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


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html', natija_usd=None)


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'GET':
        return render_template('index.html', natija_usd=None)

    natija_usd = None
    hisoblangan_bmi = None
    bmi_holat = None

    if request.method == 'POST':
        yosh = request.form.get('yosh', type=float)
        boy_sm = request.form.get('height', type=float)
        vazn_kg = request.form.get('weight', type=float)
        farzandlar_soni = request.form.get('children', type=int)
        ayolmi = request.form.get('gender') == '1'
        chekadimi = request.form.get('chekadimi') == '1'
        hudud = request.form.get('hudud')

        # BMI hisoblash
        if boy_sm and vazn_kg and boy_sm > 0:
            boy_m = boy_sm / 100.0
            hisoblangan_bmi = vazn_kg / (boy_m ** 2)

        if hisoblangan_bmi is not None:
            if hisoblangan_bmi < 18.5:
                bmi_holat = "Vazn yetarsiz (Underweight)"
            elif hisoblangan_bmi < 25:
                bmi_holat = "Me'yoriy / Sog'lom (Normal)"
            elif hisoblangan_bmi < 30:
                bmi_holat = "Ortiqcha vazn (Overweight)"
            else:
                bmi_holat = "Semizlik (Obese)"

            natija_usd = sugurta_narxini_bashorat_qil(
                yosh=yosh,
                bmi=hisoblangan_bmi,
                farzandlar_soni=farzandlar_soni,
                ayolmi=ayolmi,
                chekadimi=chekadimi,
                hudud=hudud,
                model=eng_yaxshi_model,
                scaler=scaler,
                ustunlar=ustunlar_tartibi
            )

            natija_usd = round(float(natija_usd), 2)
            hisoblangan_bmi = round(hisoblangan_bmi, 1)

    return render_template(
        'index.html',
        natija_usd=f"{natija_usd:,.2f}".replace(',', ' ') if natija_usd else None,
        bmi=hisoblangan_bmi,
        bmi_holat=bmi_holat
    )


if __name__ == "__main__":
    app.run(debug=True)