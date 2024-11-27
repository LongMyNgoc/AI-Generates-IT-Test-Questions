import spacy
from collections import Counter
import random

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def generate_mcqs(text, num_questions=5):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract sentences from the text
    sentences = [sent.text for sent in doc.sents]

    # Randomly select sentences to form questions
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))

    # Initialize list to store generated MCQs
    mcqs = []

    # Generate MCQs for each selected sentence
    for sentence in selected_sentences:
        # Process the sentence with spaCy
        sent_doc = nlp(sentence)

        # Extract entities (nouns) from the sentence
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]

        # Ensure there are enough nouns to generate MCQs
        if len(nouns) < 2:
            continue

        # Count the occurrence of each noun
        noun_counts = Counter(nouns)

        # Select the most common noun as the subject of the question
        if noun_counts:
            subject = noun_counts.most_common(1)[0][0]

            # Generate the question stem
            question_stem = sentence.replace(subject, "__________")

            # Generate answer choices
            answer_choices = [subject]

            # Add some random words from the text as distractors
            for _ in range(3):
                distractor = random.choice(list(set(nouns) - set([subject])))
                answer_choices.append(distractor)

            # Shuffle the answer choices
            random.shuffle(answer_choices)

            # Append the generated MCQ to the list
            correct_answer = chr(64 + answer_choices.index(subject) + 1)  # Convert index to letter
            mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs
print('finish')

tech_text = """
Object-oriented programming aims to implement real-world entities like inheritance, hiding, polymorphism, etc. in programming. The main aim of OOPs is to bind together the data and the functions that operate on them so that no other part of the code can access this data except that function.
A Class is a user-defined blueprint or prototype from which objects are created. It represents the set of properties or methods that are common to all objects of one type. Using classes, you can create multiple objects with the same behavior instead of writing their code multiple times. This includes classes for objects occurring more than once in your code. In general, class declarations can include these components in order: 

Modifiers: A class can be public or have default access (Refer to this for details).
Class name: The class name should begin with the initial letter capitalized by convention.
Body: The class body is surrounded by braces, { }.
An Object is a basic unit of Object-Oriented Programming that represents real-life entities. A typical Java program creates many objects, which as you know, interact by invoking methods. The objects are what perform your code, they are the part of your code visible to the viewer/user. An object mainly consists of: 

State: It is represented by the attributes of an object. It also reflects the properties of an object.
Behavior: It is represented by the methods of an object. It also reflects the response of an object to other objects.
Identity: It is a unique name given to an object that enables it to interact with other objects.
Method: A method is a collection of statements that perform some specific task and return the result to the caller. A method can perform some specific task without returning anything. Methods allow us to reuse the code without retyping it, which is why they are considered time savers. In Java, every method must be part of some class, which is different from languages like C, C++, and Python. 
"""
mcqs = generate_mcqs(tech_text, num_questions=10)  # Pass the selected number of questions
# Ensure each MCQ is formatted correctly as (question_stem, answer_choices, correct_answer)
mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]

for question in mcqs_with_index:
    print("Question", question[0], ":", question[1][0])
    print("Options:")
    options = question[1][1]
    for i, option in enumerate(options):
        print(f"{chr(97 + i)}) {option}")
    print("Correct Answer:", question[1][2])
    print("\n")