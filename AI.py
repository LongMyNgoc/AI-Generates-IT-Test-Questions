from transformers import pipeline
import random
import pandas as pd

# Tải mô hình QA
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Hàm sinh đáp án nhiễu từ ngữ cảnh
def generate_distractors(answer, context, num_distractors=3):
    """
    Sinh các đáp án nhiễu từ ngữ cảnh.
    - answer: đáp án chính xác.
    - context: đoạn văn cung cấp.
    - num_distractors: số lượng đáp án nhiễu.
    """
    # Tách từ và loại bỏ từ đã trong đáp án chính xác
    words = list(set(context.split()) - set(answer.split()))
    # Lấy ngẫu nhiên num_distractors từ
    distractors = random.sample(words, min(num_distractors, len(words)))
    return distractors

# Đọc file CSV chứa đoạn văn lĩnh vực CNTT
data = pd.read_csv("data.csv")

# Sinh câu hỏi trắc nghiệm
for idx, row in data.iterrows():
    context = row['context']
    questions = row['questions'].split(";")  # Các câu hỏi được phân cách bằng dấu chấm phẩy (;)

    print(f"\n=== Generating questions for context {idx + 1} ===")
    print(f"Context: {context}\n")
    
    for question in questions:
        # Lấy đáp án từ mô hình QA
        result = qa_pipeline(question=question.strip(), context=context)
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
