from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV
data = pd.read_csv("conv.csv")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    chat = ""

    if request.method == "POST":

        old_chat = request.form.get("chat", "")

        qts = request.form.get("qts", "")
        qts = qts.strip().lower()

        # vectorization
        texts = [qts] + data["question"].str.lower().tolist()

        cv = CountVectorizer()

        vector = cv.fit_transform(texts)

        cs = cosine_similarity(vector)

        score = cs[0][1:]

        data["score"] = score * 100

        result = data.sort_values(by="score", ascending=False)

        result = result[result.score > 10]


        if len(result) == 0:

            ans = "Sorry, I don't know that yet."

        else:

            ans = result.head(1)["answer"].values[0]


        # Chat bubble format
        new_chat = f"""
        <div class='user'>You: {qts}</div>
        <div class='bot'>Aishwarya AI: {ans}</div>
        """

        chat = old_chat + new_chat


        return render_template("home.html", chat=chat)


    return render_template("home.html", chat="")



if __name__ == "__main__":

   #app.run(debug=True, use_reloader=True)