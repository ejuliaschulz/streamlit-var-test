import pickle
from statsmodels.tsa.vector_ar.var_model import VAR
import numpy as np
import pandas as pd
import streamlit as st

def main():
    #import data
    with open("data_in.pickle", 'rb') as input:
        df = pickle.load(input)

    #Build the model
    model = VAR(df, freq='MS')

    aic, bic, fpe, hqic = [], [], [], []

    o_range = np.arange(1, 11)
    for i in o_range:
        result = model.fit(i)
        try:
            aic.append(result.aic)
            bic.append(result.bic)
            fpe.append(result.fpe)
            hqic.append(result.hqic)
        except Exception as e:
            print("for order = " + str(i))
            print(f"Unexpected Error {e=}, {type(e)=}")
            aic.append(None)
            bic.append(None)
            fpe.append(None)
            hqic.append(None)

    lags_metrics_df = pd.DataFrame({'AIC': aic,
                                    'BIC': bic,
                                    'HQIC': hqic,
                                    'FPE': fpe},
                                   index=range(1, len(aic) + 1))


    best_model_order = lags_metrics_df.idxmin(axis=0)['AIC']

    st.set_page_config(
        page_title="Test",
        layout="wide",  #centered was a bit too small
        initial_sidebar_state="auto",
    )

    st.write(df)
    st.write("Best model order found: " + str(best_model_order))
     st.write(lags_metrics_df)

if __name__ == "__main__":
    main()
