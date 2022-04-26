import streamlit as st
import pandas as pd
import argparse
import requests
import datetime
import json
import requests
import csv
import os
import subprocess 
import re
from sentence_transformers import SentenceTransformer, util
import itertools
from st_aggrid import AgGrid, JsCode, GridUpdateMode
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def rerun():
    raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))

@st.cache
def generate_questions(args,text):
    body = {"text":text}

    res = requests.post(args.question_generation_endpoint, json=body)

    if res.status_code != 200:
        raise RuntimeError("Something went wrong while calling the question answering module, try again later")

    response_json = res.json()

    return [(response_json['t5']['question'], response_json['t5']['answer'], response_json['t5']['score'], response_json['t5']['answer_score']), 
    (response_json['bart']['question'], response_json['bart']['answer'], response_json['bart']['score'], response_json['bart']['answer_score'])]

def print_title(title):
     st.markdown("""
        <img style="float: right; width: 200px;" src="https://esatde.expertcustomers.ai/images/EAI PRIMARY TRANSPARENT.png">
        <img style="float: right; width: 200px;" src="https://esatde.expertcustomers.ai/images/ESA_logo_2020_Deep-1024x643.jpg">
        <h1 style="position: relative">"""+title+"""</h1>
        """, unsafe_allow_html=True)

def get_paragraphs(texts):
    ## Remove headers and footers and merge all pages
    text_pages = []
    p = re.compile(r'\n \n \nPage \d+/\d+ \n[\w \-\n]+\nDate \d+/\d+/\d+  Issue \d  Rev \d \nESA UNCLASSIFIED \- For (ESA )*Official Use( )*(Only )*\n')
    for text in texts:
        text = p.sub('',text)
        text_pages.append(text[:-1])
    doc = "".join(text_pages)

    p1 = re.compile(r'(\n *\d+(?:\.\d+)* [A-Za-z ]+)(\n)',re.DOTALL)
    doc = p1.sub('\\1**Special character**',doc)

    p1 = re.compile(r'(\*\*Special character\*\* *\d+(?:\.\d+)* [A-Za-z ]+)(\n)',re.DOTALL)
    doc = p1.sub('\\1**Special character**',doc)
    
    p2 = re.compile(r'(\n(?!(?:\d|APPENDIX| *\n)))',re.DOTALL)
    oc = p2.sub('',doc)

    p3 = re.compile(r'\*\*Special character\*\*',re.DOTALL)
    doc = p3.sub('\n',doc)

    paragraphs = re.split(r'^APPENDIX|^(?=\d+(?:\.\d+)* [A-Za-z ]+\b)', doc, flags=re.MULTILINE)
    r = re.compile(r'^\d(?!.*\.{5})')
    filtered_paragraphs = list(filter(r.match,paragraphs))

    return filtered_paragraphs

def question_generation_demo_v2(args):
    print_title("Space Quality Quiz")
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'download' not in st.session_state:
        st.session_state['download'] = None    
    if 'test-document' not in st.session_state:
        st.session_state['test-document'] = None 

    st.markdown("Code available at [SpaceQQuiz Github repository](https://github.com/expertailab/SpaceQQuiz).")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file:
        st.session_state['test-document'] = False

    use_test_document = st.button("Use test document")
    if use_test_document:
        record = st.uploaded_file_manager.UploadedFileRec('','RD-2-OCCand-ESTRACK-operation-manual.pdf','',b'')
        uploaded_file = st.uploaded_file_manager.UploadedFile(record)

    st.markdown("Test document link: [RD-2-OCCand-ESTRACK-operation-manual.pdf](https://esastar-publication.sso.esa.int/api/filemanagement/download?url=emits.sso.esa.int/emits-doc/ESOC/5189/RD-2-OCCand-ESTRACK-operation-manual.pdf)")


    if uploaded_file is not None or st.session_state['test-document']:

        if (not st.session_state['test-document']) and (not st.session_state['uploaded_file'] or (st.session_state['uploaded_file'].name != uploaded_file.name)):
            my_bar = st.progress(0)
            st.session_state['uploaded_file'] = uploaded_file

            if use_test_document:
                st.session_state['test-document'] = True
            else:    
                with open(os.path.join("uploads",uploaded_file.name),"wb") as f:
                    f.write((uploaded_file).getbuffer())
            my_bar.progress(0.33)

            result = subprocess.run("java -jar pdf-extractor-0.2.0-SNAPSHOT-standalone.jar -c "+("esa-tde-docs_config.json" if not use_test_document else "esa-tde-docs-test_config.json"), shell=True)
            my_bar.progress(0.66)

            with open(os.path.join("target","esa-tde-txts-out",uploaded_file.name+".json"), encoding="utf8") as json_file:
                json_text = json.load(json_file)
                st.session_state.snipets = get_paragraphs([page_text["gresladix/text"] for page_text in json_text["gresladix/page-texts"]])
            my_bar.progress(1.0)

            try:
                os.remove(os.path.join("uploads",uploaded_file.name)) 
            except OSError:
                pass
            my_bar.empty()

        form = st.form(key='my_form')
        form.write('Select the sections you want to use to generate questions by clicking on the checkboxes, when finished, click on the "Generate questions" button.')
        select_section = {i:form.checkbox(snipet.split('\n')[0], value=False,key=id(i)) for i,snipet in enumerate(st.session_state.snipets)}
        generate_questions_button = form.form_submit_button('Generate questions')

        if generate_questions_button and np.any(list(select_section.values())):
            my_bar = st.progress(0)
            df = pd.DataFrame(columns=['question', 'answer', 'model', 'paragraph'])
            snipets = [text for i,text in enumerate(st.session_state.snipets) if len(text) > 100 and select_section[i]]

            snipets_with_title = []
            snipets_without_title = []
            titles = [snipet.split('\n')[0] for snipet in snipets]
            for index, snipet in enumerate(snipets):
                paragraphs = re.split(r'\.[ \n]*([A-Z][a-z](?:[^•\.]|\.(?! (?:[A-Z][a-z]|\n)))*•[^\n]*\n)', snipet, flags=re.MULTILINE)
                for paragraph_index, paragraph in enumerate(paragraphs):
                    snipets_with_title += [titles[index]+'\n'+text if paragraph_index==0 and j == 0 else text for j,text in enumerate(['\n'.join([sublist[0] if sublist[0] else '', sublist[1] if sublist[1] else '']) for sublist in list(itertools.zip_longest(*[iter(paragraph.split('\n')[1:] if paragraph_index==0 else paragraph.split('\n')[0:])]*2))]) if text.strip()]
                    snipets_without_title += [text  for text in ['\n'.join([sublist[0] if sublist[0] else '', sublist[1] if sublist[1] else '']) for sublist in list(itertools.zip_longest(*[iter(paragraph.split('\n')[1:] if paragraph_index==0 else paragraph.split('\n')[0:] )]*2))] if text.strip()]
            
            for i,text in enumerate(snipets_without_title):
                st.session_state.context = text ## Save the context that was used to generate question
                res = generate_questions(args,st.session_state.context)
                st.session_state.question_t5, st.session_state.answer_t5, st.session_state.score_t5, st.session_state.answer_t5_score = res[0]
                st.session_state.question_bart, st.session_state.answer_bart, st.session_state.score_bart, st.session_state.answer_bart_score = res[1]
                if st.session_state.answer_t5.strip() == '' or st.session_state.question_t5[-1]!='?':
                    st.session_state.question_t5 = ''
                    st.session_state.answer_t5 = ''
                if st.session_state.answer_bart.strip() == '' or st.session_state.question_bart[-1]!='?':
                    st.session_state.question_bart = ''
                    st.session_state.answer_bart = ''

                df=df.append({'question':st.session_state.question_t5, 'question_score':round(st.session_state.score_t5,4), 'answer':st.session_state.answer_t5, 'answer_score': round(st.session_state.answer_t5_score,4), 'model':'t5','paragraph': snipets_with_title[i] }, ignore_index=True)
                df=df.append({'question':st.session_state.question_bart, 'question_score': round(st.session_state.score_bart,4), 'answer':st.session_state.answer_bart, 'answer_score': round(st.session_state.answer_bart_score,4), 'model':'bart', 'paragraph': snipets_with_title[i]}, ignore_index=True)

                my_bar.progress(float((i+1)/len(snipets_without_title)))

            sentences = df['question'].tolist()
            df = delete_repeated_questions(sentences,df)
            # sentences_bart = df['question_bart'].tolist()
            # df = delete_repeated_questions(sentences_bart,df,'bart')
            # sentences_bart = df['question_bart'].tolist()
            # df = delete_repeated_questions(sentences_t5,df,'bart', sentences2=sentences_bart)
            df = df[(df['question'] != '') | (df['question'] != '')]
            my_bar.empty()
            st.session_state['df'] = df
            
        if st.session_state['df'] is not None:
            grid_options = {
                "columnDefs": [
                    {"field": "question", "checkboxSelection": True},
                    {"field": "question_score", "type":["numericColumn", "numberColumnFilter"]},
                    {"field": "answer"},
                    {"field": "answer_score", "type":["numericColumn", "numberColumnFilter"]},
                    {"field": "model"},
                    {"field": "paragraph"}
                ],
                "rowSelection":"multiple",
                "suppressRowClickSelection":"true"
            }      
            with st.form("my_form2"):
                st.write('Select the questions you want to include in the quiz by clicking on the checkboxes, when you are done, click on the "Generate document" button.')
                grid_return = AgGrid(st.session_state['df'], grid_options, update_mode=GridUpdateMode.MODEL_CHANGED)
                new_df = grid_return['selected_rows']
                generate_document = st.form_submit_button("Generate document")
                if generate_document:
                    st.session_state['download'] = True
            if st.session_state['download'] and len(new_df)>0:
                include_scores = st.checkbox('Include scores', value=False)
                if include_scores:
                    new_df = pd.DataFrame(new_df)
                else:
                    new_df = pd.DataFrame(new_df)[['question','answer','paragraph']]
                csv = convert_df(new_df[['question','answer','paragraph']])
                st.download_button(label="Download data as CSV", data=csv, file_name='document.csv', mime='text/csv')
                html = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Quiz</title>
  </head>
  <body>
    <h1>Questions:</h1>"""
                for index, row in new_df.iterrows():
                    html += "<p>" + str(index+1) + ". " + row["question"] + "<p>" 
                html += "<h1>Questions + answers + paragraph:</h1>"

                for index, row in new_df.iterrows():
                    html += "<p> <b>"+ str(index+1) + ". Question:</b> " + row["question"] + (" ("+str(row["question_score"])+")" if include_scores else "") + "<p>" 
                    html += "<p> <b>Answer:</b> " + row["answer"] + (" ("+str(row["answer_score"])+")" if include_scores else "") + "<p>" 
                    html += "<p> <b>Paragraph:</b> "
                    matches = []

                    row["paragraph"] = row["paragraph"].replace("\n", "<br/>") 

                    if row["answer"].strip():
                        for match in re.finditer(re.escape(row["answer"]), row["paragraph"]):
                            matches.append(match)
            
                    # reverse sorting
                    matches = sorted(matches, key = lambda x: x.start(), reverse=True)
                    if len(matches)==0:
                        html += row["paragraph"]
                    else:
                        for match in matches[:1]:
                            print(match)
                            html += row["paragraph"][:match.start()] +\
                                "<span style='color:red;'> %s </span>" % row["paragraph"][match.start():match.end()] +\
                                row["paragraph"][match.end():]

                    html += "<p>"
                    html += "<br />" 
                html+="""</body>
</html>"""
                st.download_button(label="Download data as HTML", data=html, file_name='document.html', mime='text/html')
            #display_highlighted_words(df,'')
            #st.table(df)
    st.session_state['uploaded_file'] = uploaded_file

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def delete_repeated_questions(sentences, df, sentences2=None):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    if sentences2:
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings2)
    else:
        cosine_scores = util.cos_sim(embeddings, embeddings)
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            if cosine_scores[i][j] > 0.8: ## Discard repeated questions
                df['question'][j] = ''
                df['answer'][j] = ''
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    for pair in pairs:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))

    return df

def about():
    print_title("About")
    st.write("""
    ### Space Quality Quiz
    SpaceQQuiz is a system to generate quizzes, a common resource to evaluate training sessions, out of quality procedure documents in the Space domain. Our system leverages state of the art auto-regressive models like [T5](https://arxiv.org/pdf/1910.10683.pdf) and [BART](https://arxiv.org/abs/1910.13461) to generate questions, and a [RoBERTa](https://arxiv.org/abs/1907.11692) model to extract answer for the questions, thus verifying their suitability.

    If you have any doubt or suggestion, please send it to [Andrés García](mailto:agarcia@expert.ai), [Cristian Berrío](mailto:cberrio@expert.ai) or [Jose Manuel Gómez](mailto:jmgomez@expert.ai).
    """)

def run_app(args, session_state=None):

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    menu_opts = {
        1: "Space Quality Quiz",
        2: "About"
    }

    menu_box = st.sidebar.selectbox('MENU', (
        menu_opts[1],
        menu_opts[2],
    ))

    if menu_box == menu_opts[1]:
        question_generation_demo_v2(args)

    if menu_box == menu_opts[2]:
        about()

