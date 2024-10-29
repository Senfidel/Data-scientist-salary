from fastapi import FastAPI, Form, Request
import uvicorn
import joblib
import pandas as pd
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

# Configuration des templates Jinja2
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Charger le fichier CSV avec les caractéristiques
df = pd.read_csv("salaries_cleaned.csv")  # Remplacez "data.csv" par le chemin de votre fichier CSV

# Charger le modèle préalablement entraîné
model = joblib.load("SVR_pipeline.pkl")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/get_unique_values/{field_name}")
def get_unique_values(field_name: str):
    """Route pour obtenir les valeurs uniques des champs dynamiques."""
    # Vérifiez que le champ demandé existe bien dans le DataFrame
    if field_name in df.columns:
        unique_values = df[field_name].unique().tolist()
        return {"unique_values": unique_values}
    return {"error": "Field not found"}, 404


@app.post("/score", response_class=HTMLResponse)
def score(request: Request,
          work_year: int = Form(...),
          salary_currency: str = Form(...),
          experience_level: str = Form(...),
          job_title: str = Form(...),
          employee_residence: str = Form(...),
          remote_ratio: int = Form(...),
          company_location: str = Form(...),
          company_size: str = Form(...),
          employment_type: str = Form(...)):
    
    # Vérifier les données reçues (pour débogage)
    print("Received data:", {
        "work_year": work_year,
        "salary_currency": salary_currency,
        "experience_level": experience_level,
        "job_title": job_title,
        "employee_residence": employee_residence,
        "remote_ratio": remote_ratio,
        "company_location": company_location,
        "company_size": company_size,
        "employment_type": employment_type
    })
    
    # Création des données sous forme de DataFrame pour le modèle
    try:
        data = pd.DataFrame([{
            "work_year": work_year,
            "salary_currency": salary_currency,
            "experience_level": experience_level,
            "job_title": job_title,
            "employee_residence": employee_residence,
            "remote_ratio": remote_ratio,
            "company_location": company_location,
            "company_size": company_size,
            "employment_type": employment_type
        }])
    except Exception as e:
        print("Error creating DataFrame:", e)
        return templates.TemplateResponse("home.html", {
            "request": request,
            "prediction_text": "Erreur lors de la création des données pour le modèle."
        })

    # Prédire avec le modèle
    try:
        resultat = model.predict(data)[0]
        prediction_usd = np.expm1(resultat)  # Inverse de la transformation logarithmique
        prediction_eur = prediction_usd * 0.85  # Conversion en EUR
    except Exception as e:
        print("Error during prediction:", e)
        return templates.TemplateResponse("home.html", {
            "request": request,
            "prediction_text": "Erreur lors de la prédiction du modèle."
        })

    # Passer le résultat au template
    return templates.TemplateResponse("home.html", {
        "request": request,
        "prediction_text": f"Le salaire estimé est de {prediction_eur:.2f} EUR"
    })

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)




# Personnalisation du format pour enrichir la documentation Swagger
#class CaracteristiquesFormat(BaseModel):
#    work_year: int
#   salary_currency: str
#    experience_level: str
#    job_title: str
#    employee_residence: str
 #   remote_ratio: int
#    company_location: str
 #   company_size: str
 #   employment_type: str

# Charger le modèle préalablement entraîné
#model = joblib.load("SVR_pipeline.pkl")

#@app.post("/score")
#def score(caracteristiques: CaracteristiquesFormat):
#    normalized_df = pd.json_normalize(caracteristiques.model_dump())
#    resultat = model.predict(normalized_df)
#    prediction_usd = np.expm1(resultat)
#   prediction_eur = prediction_usd * 0.85
#    resultat_list = prediction_eur .tolist() if isinstance(prediction_eur , np.ndarray) else [prediction_eur]
 #   return f"le salaire est estimé à {resultat_list}"

#if __name__ == '__main__':
#    uvicorn.run(app, host="0.0.0.0", port=8001)