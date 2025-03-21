import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Flask 앱 생성
app = Flask(__name__)

# ✅ 한국어 문장 예제 데이터셋
corpus = [
    "나는 너를 사랑해",
    "나는 코딩을 좋아해",
    "너는 나를 좋아해",
    "너는 파이썬을 공부해",
    "우리는 인공지능을 연구해",
    "딥러닝은 재미있어",
    "파이썬은 강력해",
    "나는 자연어처리를 공부해",
]

# ✅ 단어 사전 만들기
word_list = list(set(" ".join(corpus).split()))
word_dict = {w: i for i, w in enumerate(word_list)}
idx_dict = {i: w for w, i in word_dict.items()}


# ✅ 모델 정의
class RNNTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(RNNTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # 단어 임베딩
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 마지막 시점의 RNN 출력을 사용
        return out


# ✅ 저장된 모델 불러오기 함수
def load_model(model_path, vocab_size, embed_size, hidden_size, num_classes):
    model = RNNTextModel(vocab_size, embed_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


# ✅ 모델 로드
model_path = "model/rnn_korean_model.pth"
model = load_model(model_path, len(word_dict), 10, 16, len(word_dict))


# ✅ 문장 예측 함수
def predict_next_word(sentence):
    words = sentence.split()
    input_seq = [word_dict[w] for w in words if w in word_dict]
    max_len = max(len(seq) for seq in [[word_dict[w] for w in s.split()] for s in corpus])
    input_padded = input_seq + [0] * (max_len - len(input_seq))
    input_tensor = torch.tensor([input_padded], dtype=torch.long)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    predicted_word = idx_dict[predicted_idx]
    return predicted_word, confidence


# ✅ 웹페이지 렌더링
@app.route("/")
def index():
    return render_template("index.html")


# ✅ 예측 API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sentence = data.get("sentence", "")
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    predicted_word, confidence = predict_next_word(sentence)
    return jsonify({"predicted_word": predicted_word, "confidence": round(confidence * 100, 2)})


# ✅ Flask 서버 실행
if __name__ == "__main__":
    app.run(debug=True)

