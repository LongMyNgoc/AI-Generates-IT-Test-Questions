from transformers import pipeline
import random
import pandas as pd

# Tải pipeline cho QA và sinh câu hỏi
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
question_generation_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

# Hàm sinh câu hỏi từ ngữ cảnh
def generate_questions(context, max_questions=3):
    """
    Sinh câu hỏi từ đoạn văn.
    - context: Đoạn văn cung cấp.
    - max_questions: Số lượng câu hỏi tối đa cần sinh.
    """
    input_text = f"generate questions: {context}"
    questions = question_generation_pipeline(input_text, max_length=64, num_return_sequences=1)
    return [q['generated_text'] for q in questions]

# Hàm sinh đáp án nhiễu từ ngữ cảnh
def generate_distractors(answer, context, num_distractors=3):
    """
    Sinh các đáp án nhiễu từ ngữ cảnh.
    - answer: Đáp án chính xác.
    - context: Đoạn văn cung cấp.
    - num_distractors: Số lượng đáp án nhiễu.
    """
    words = list(set(context.split()) - set(answer.split()))
    distractors = random.sample(words, min(num_distractors, len(words)))
    return distractors

# Đọc file CSV chứa đoạn văn lĩnh vực CNTT
data = pd.read_csv("data.csv")

# Sinh câu hỏi trắc nghiệm
for idx, row in data.iterrows():
    context = row['context']
    
    print(f"\n=== Generating questions for context {idx + 1} ===")
    print(f"Context: {context}\n")
    
    # Tạo câu hỏi từ đoạn văn
    questions = generate_questions(context, max_questions=3)

    for question in questions:
        # Lấy đáp án từ mô hình QA
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
        
        # Sinh đáp án nhiễu
        distractors = generate_distractors(answer, context)
        options = [answer] + distractors
        random.shuffle(options)  # Xáo trộn thứ tự đáp án
        
        # In câu hỏi và các đáp án
        print(f"Question: {question}")
        for i, option in enumerate(options):
            print(f"{chr(65 + i)}. {option}")
        print()
