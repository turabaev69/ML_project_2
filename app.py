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

def sonni_soz_bilan_ifodalash(son):
    if son == 0:
        return "nol"

    biraklar = ["", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
    onliklar = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]

    def uchlikni_sozga_aylantirish(n):
        natija = ""
        yuzlik = n // 100
        onlik = (n % 100) // 10
        birak = n % 10

        if yuzlik > 0:
            if yuzlik == 1:
                natija += "yuz "
            else:
                natija += biraklar[yuzlik] + " yuz "

        if onlik > 0:
            natija += onliklar[onlik] + " "

        if birak > 0:
            natija += biraklar[birak] + " "

        return natija.strip()

    if son < 0:
        return "minus " + sonni_soz_bilan_ifodalash(-son)

    daraja_nomlari = ["", "ming", "million", "milliard", "trillion"]
    uchliklar = []
    
    temp = int(son)
    while temp > 0:
        uchliklar.append(temp % 1000)
        temp //= 1000

    sozlar = []
    for i, uchlik in enumerate(uchliklar):
        if uchlik > 0:
            uchlik_matni = uchlikni_sozga_aylantirish(uchlik)
            if i > 0:
                if i == 1 and uchlik == 1:
                    sozlar.append("ming")
                else:
                    sozlar.append(uchlik_matni + " " + daraja_nomlari[i])
            else:
                sozlar.append(uchlik_matni)

    return " ".join(reversed(sozlar)).strip()

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
    return render_template('index.html', natija_usd=None, natija_uzs=None, natija_uzs_soz=None)

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == 'GET':
        return render_template('index.html', natija_usd=None, natija_uzs=None, natija_uzs_soz=None)

    natija_usd = None
    natija_uzs = None
    natija_uzs_soz = None
    hisoblangan_bmi = None
    bmi_holat = None
    yosh = None
    boy_sm = None
    vazn_kg = None
    farzandlar_soni = None
    ayolmi = False
    chekadimi = False
    hudud = 'northeast'

    if request.method == 'POST':
        yosh = request.form.get('yosh', type=float)
        boy_sm = request.form.get('height', type=float)
        vazn_kg = request.form.get('weight', type=float)
        farzandlar_soni = request.form.get('children', type=int)
        ayolmi = request.form.get('gender') == '1'
        chekadimi = request.form.get('chekadimi') == '1'
        hudud = request.form.get('hudud')

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
            
            kurs_usd_uzs = 12600
            summa_uzs = natija_usd * kurs_usd_uzs
            natija_uzs = round(summa_uzs, -3)

            natija_uzs_soz = sonni_soz_bilan_ifodalash(int(natija_uzs))

            hisoblangan_bmi = round(hisoblangan_bmi, 1)

    return render_template(
        'index.html',
        natija_usd=f"{natija_usd:,.2f}".replace(',', ' ') if natija_usd else None,
        natija_uzs=f"{natija_uzs:,.0f}".replace(',', ' ') if natija_uzs else None,
        natija_uzs_soz=natija_uzs_soz.capitalize() if natija_uzs_soz else None,
        bmi=hisoblangan_bmi,
        bmi_holat=bmi_holat,
        yosh=yosh,
        boy_sm=boy_sm,
        vazn_kg=vazn_kg,
        farzandlar_soni=farzandlar_soni,
        ayolmi=ayolmi,
        chekadimi=chekadimi,
        hudud=hudud
    )

if __name__ == "__main__":
    app.run(debug=True)