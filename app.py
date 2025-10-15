import streamlit as st
import pandas as pd, numpy as np, math as mt, joblib, os
from scipy.sparse import csr_matrix, vstack
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors


st.set_page_config(layout="wide", page_title="Polypharmacy Side Effect Predictor", page_icon="🧬")
st.title("Polypharmacy Side Effect Predictor")
st.markdown("##### A Double ML Side Effect Predictor Based on Physicochemical Properties of Any Two Drugs")
st.markdown("####")
smi1 = st.text_area(
    "Drug 1 [Canonical SMILES/InChI Expected]"
)
smi2 = st.text_area(
    "Drug 2 [Canonical SMILES/InChI Expected]"
)

if st.button("Predict Side Effect", type="primary"):
    # st.write(smi1)
    # st.write(smi2)
    # try:
    with st.spinner('Your query submitted successfully. Please wait...'):
        st.write(os.getcwd().replace("/models", ""))
        os.chdir(os.getcwd().replace("/models", ""))
        # os.chdir(r'C:\mtech_bioinfo\MTECH_Course\sem3\minor project\pred app')
        def calculate_descriptors(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Invalid SMILES"
            descriptor_values = [descriptor(mol) for name, descriptor in Descriptors._descList]
            logp = Crippen.MolLogP(mol)
            des_ = descriptor_values + [logp]
            return [0 if np.isnan(x) else x for x in des_]
        def inchi_calculate_descriptors(smiles):
            mol = Chem.MolFromInchi(smiles)
            if mol is None:
                return "Invalid SMILES"
            descriptor_values = [descriptor(mol) for name, descriptor in Descriptors._descList]
            logp = Crippen.MolLogP(mol)
            des_ = descriptor_values + [logp]
            return [0 if np.isnan(x) else x for x in des_]
        d1_d2_descriptor_names = ["d1_d2_abs_diff_"+desc[0] for desc in Descriptors._descList] + ['d1_d2_abs_diff_LogP']
        uniqueSideEffects = "".join(open("uniqueSideEffects").readlines()).split("\n")
        mapppingSE = {}
        c = 0
        for sideEffect in uniqueSideEffects:
            mapppingSE.update({sideEffect: c})
            c+=1
        # smi1, smi2 = 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C', 'CC(=CC=CC=C(C)C=CC=C(C)C(=O)OC1C(C(C(C(O1)COC2C(C(C(C(O2)CO)O)O)O)O)O)O)C=CC=C(C)C(=O)OC3C(C(C(C(O3)COC4C(C(C(C(O4)CO)O)O)O)O)O)O'
        # s1_s2_abs_diff = np.round(np.abs(np.subtract(np.abs(calculate_descriptors(smi1)), np.abs(calculate_descriptors(smi2)))), decimals=6)
        # st.write(smi1[:5])
        s1_s2_abs_diff = []
        if smi1[:5] == "InChI":
            s1_s2_abs_diff = np.round(np.abs(np.subtract(np.abs(inchi_calculate_descriptors(smi1)), np.abs(inchi_calculate_descriptors(smi1)))), decimals=6)
        if smi1[:5] != "InChI":
            try: 
                s1_s2_abs_diff = np.round(np.abs(np.subtract(np.abs(calculate_descriptors(smi1)), np.abs(calculate_descriptors(smi2)))), decimals=6)
            except:
                st.warning("Unable to parse SMILES. Please try with InChI!")
        if len(s1_s2_abs_diff) > 0:
            col_header = d1_d2_descriptor_names+uniqueSideEffects
            x = csr_matrix((0, len(col_header)-1))
            c = 0
            for lable in range(len(mapppingSE)):
                seMat = [0] * len(uniqueSideEffects)
                seMat[lable] = 1
                merged_array = np.hstack((s1_s2_abs_diff, np.array(seMat[:-1])))
                new_row = csr_matrix(merged_array)
                x = vstack([x.astype(float), new_row])
                c+=1
            model = joblib.load('ppmodel.joblib')
            model_out = model.predict_proba(x)
            mapppingSE = {}
            c = 0
            for sideEffect in uniqueSideEffects:
                mapppingSE.update({c : sideEffect})
                c+=1
            c = 0
            outScore = {}
            for i in model_out:
                outScore.update({mapppingSE[c]: list(i)[c]})
                c+=1
            # sortedOutScore = sorted(outScore.values(), reverse=True)
            # for i in outScore:
            #     print(i, outScore[i])
            # print(outScore)
            col_header = ["d1_d2_abs_diff_"+desc[0] for desc in Descriptors._descList] + ['d1_d2_abs_diff_LogP']
            inpX = csr_matrix((0, len(col_header)))
            inpX = vstack([inpX.astype(float), s1_s2_abs_diff])
            # save_npz(f'inpX.npz', x)
            st.write(os.getcwd())
            os.chdir(rf'{os.getcwd()}/models')
            # os.chdir(r'C:\mtech_bioinfo\MTECH_Course\sem3\minor project\pred app\models')
            predSe = []
            progress_text = "Prediction in progress. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            k = 0
            for i, j in zip(range(len(mapppingSE)), outScore.keys()):
                model = joblib.load(f'model_{i}.joblib')
                specificModelPredict = model.predict(inpX)[0]
                # print(i)
                if i%131 == 0:
                    my_bar.progress(k+9, text=progress_text)
                    k+=9
                #     st.text("".join(["=>"]*(i%137)))
                # if specificModelPredict != -1:
                #     predSe.append({
                #         "Side Effect": j,
                #         "Probability": outScore[mapppingSE[i]]
                #     })
                try:
                    specificModelPredict = model.predict(inpX)[0]
                    if specificModelPredict != -1:
                        predSe.append({
                            "Side Effect": j,
                            "Probability": str(outScore[mapppingSE[i]])
                        })
                except:
                    if outScore[mapppingSE[i]] >= 0.5:
                        predSe.append({
                                "Side Effect": j,
                                "Probability": str(outScore[mapppingSE[i]])
                            })
                    # print(mapppingSE[i], model.predict(inpX)[0], j, outScore[mapppingSE[i]])
                    # print(mapppingSE[i], model.predict(inpX)[0], j, outScore[mapppingSE[i]])
            my_bar.empty()
            # yAxis = list(np.round(np.arange(0.0, 1.1, 0.1), decimals=2))
            chartDf = pd.DataFrame(predSe)
            dfCopy = chartDf.copy()
            dfCopy["Probability"] = dfCopy["Probability"].astype(float)
            st.write("######")
            st.write("###### Predicted Side Effect With Probabilities")
            st.table(dfCopy)
            st.write("######")
            st.write("###### Barplot of Predicted Side Effect [Hover on Bar to See the Specific Side Effect with Probabilities]")
            st.bar_chart(dfCopy, x="Side Effect", y="Probability")
            # dfCopy.to_csv("dd-se-predicted.csv", index=False)
            # @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')
            csv = convert_df(dfCopy)
            st.download_button(
                label="Download Predicted Side Effects as CSV",
                data=csv,
                file_name='dd-se-predicted.csv',
                mime='text/csv',
            )
            st.success("")
    # except: st.write("###### Please enter the canonical SMILES correctly OR the field are empty!")
else: pass
