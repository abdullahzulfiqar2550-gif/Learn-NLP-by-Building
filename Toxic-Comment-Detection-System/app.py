import re
import os
import pickle

import streamlit as st
import pandas as pd
import mysql.connector
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)

MODEL_FILE = "model.pkl"
DATASET    = "toxic_dataset.csv"

LABEL_MAP = {
    0: ("🟢 Clean","green"),
    1: ("🟡 Mildly Toxic", "orange"),
    2: ("🔴 Highly Toxic", "red"),}


def preprocess(text: str) -> str:
    text  = re.sub("[^a-zA-Z0-9]", " ", text)
    text  = text.lower().split()
    lm    = WordNetLemmatizer()
    stops = stopwords.words("english")
    stops.remove("not")
    text  = [lm.lemmatize(w) for w in text if w not in set(stops)]
    return " ".join(text)


@st.cache_resource(show_spinner="Training model on 20000 comments… (first run only)")
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            cv, clf, acc = pickle.load(f)
        return cv, clf, acc

    df      = pd.read_csv(DATASET)
    corpus  = [preprocess(c) for c in df["comment"]]
    y       = df["label"].values

    cv      = CountVectorizer()
    X       = cv.fit_transform(corpus).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)

    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump((cv, clf, acc), f)

    return cv, clf, acc


def predict(comment: str, cv, clf) -> int:
    vec = cv.transform([preprocess(comment)]).toarray()
    return int(clf.predict(vec)[0])

def get_connection():
    return mysql.connector.connect(host="mysql", user="root", password="root123", database="toxic_db")

@st.cache_resource
def init_db():
    con = mysql.connector.connect(host="mysql", user="root", password="root123")
    cur = con.cursor()
    cur.execute("CREATE DATABASE IF NOT EXISTS toxic_db")
    cur.close()
    con.close()

    con = get_connection()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            username    VARCHAR(100),
            comment     TEXT,
            label       INT,
            submitted   TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    con.commit()
    cur.close()
    con.close()


st.set_page_config(page_title="Toxic Comment Detector")
st.title("Toxic Comment Detector")

init_db()
cv, clf, acc = load_model()

st.info(f"Model accuracy on test set: **{acc * 100:.1f}%**")

menu = st.sidebar.selectbox("Menu", ["Analyse Comment", "View All Comments", "Search Comments"])


if menu == "Analyse Comment":
    st.subheader("Submit a Comment")
    username = st.text_input("Your Username")
    comment  = st.text_area("Write your comment here", height=150)

    if st.button("Analyse & Save"):
        if not username.strip():
            st.warning("Please enter your username")
        elif len(comment.strip()) < 3:
            st.warning("Comment is too short.")
        else:
            label         = predict(comment, cv, clf)
            label_text, _ = LABEL_MAP[label]

            if label == 0:
                st.success(f"Result: {label_text}")
            elif label == 1:
                st.warning(f"Result: {label_text}")
            else:
                st.error(f"Result: {label_text}")

            con = get_connection()
            cur = con.cursor()
            cur.execute("INSERT INTO comments (username, comment, label) VALUES (%s, %s, %s)",(username.strip(), comment.strip(), label))
            con.commit()
            cur.close()
            con.close()
            st.success("Comment saved to database")


elif menu == "View All Comments":
    st.subheader("All Analysed Comments")
    con = get_connection()
    df  = pd.read_sql("SELECT id, username, comment, label, submitted FROM comments ORDER BY id DESC", con)
    con.close()

    if df.empty:
        st.info("No comments yet")
    else:
        df["Toxicity"] = df["label"].apply(lambda l: LABEL_MAP[l][0])
        st.dataframe(df[["id", "username", "comment", "Toxicity", "submitted"]],use_container_width=True)
        counts = df["Toxicity"].value_counts()
        st.bar_chart(counts)


elif menu == "Search Comments":
    st.subheader("Search Comments")
    keyword = st.text_input("Search by username or keyword in comment")

    if st.button("Search"):
        if not keyword.strip():
            st.warning("Enter a search term")
        else:
            con = get_connection()
            cur = con.cursor()
            q   = f"%{keyword.strip()}%"
            cur.execute("SELECT id, username, comment, label, submitted FROM comments WHERE username LIKE %s OR comment LIKE %s",(q, q))
            rows = cur.fetchall()
            cur.close()
            con.close()

            if rows:
                df = pd.DataFrame(rows, columns=["ID", "Username", "Comment", "Label", "Submitted"])
                df["Toxicity"] = df["Label"].apply(lambda l: LABEL_MAP[l][0])
                st.dataframe(df[["ID", "Username", "Comment", "Toxicity", "Submitted"]],use_container_width=True)
            else:
                st.info("No matching comments found")
