import numpy as np
import pandas as pd
from pycaret.regression import *
from fastapi import FastAPI 


app = FastAPI()


loaded_model = load_model('my_best_pipeline')

@app.get("/get_Emissions")
async def get_distance(Federal_organization, Fiscal_year ,GHG_source ,GHG_scope,Energy_category ):

    # Ensure that this DataFrame has the same columns as the original DataFrame used in setup (df_copy in this case)
    data = pd.DataFrame([[Federal_organization, Fiscal_year ,GHG_source ,GHG_scope,Energy_category]],  # Add np.nan for missing 'GHG scope'
                columns=['Federal organization', 'Fiscal year', 'GHG source',"GHG scope", 'Energy category']) # Include 'GHG scope'

    # functional API
    predictions = predict_model(loaded_model, data=data)
    predictions.head()

    return predictions["prediction_label"]

