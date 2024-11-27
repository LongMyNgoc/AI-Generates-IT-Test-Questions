from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from collections import Counter
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)

def generate_mcqs(text, num_questions=5):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))
    mcqs = []
    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        if len(nouns) < 2:
            continue
        noun_counts = Counter(nouns)
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]
            question_stem = sentence.replace(subject, "__________")
            answer_choices = [subject]
            for _ in range(3):
                distractor = random.choice(list(set(nouns) - set([subject])))
                answer_choices.append(distractor)
            random.shuffle(answer_choices)
            correct_answer = chr(64 + answer_choices.index(subject) + 1)
            mcqs.append((question_stem, answer_choices, correct_answer))
    return mcqs

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    text = data.get('text', '')
    num_questions = data.get('num_questions', 5)
    questions = generate_mcqs(text, num_questions)
    return jsonify({"questions": questions})

if __name__ == "__main__":
    app.run(debug=True)
