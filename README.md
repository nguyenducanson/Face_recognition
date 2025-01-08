[Repo](https://github.com/mk-minchul/CVLface)

# Face Recognition 

1. Face Detection: YOLOv8
2. Face Alignment: DFA MobileNet
3. Face Recognition: AdaFace
4. Vector Database: Qdrant


### 1. Install dependents:
```bash
conda create -n cvlface python=3.10 pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/nguyenducanson/Face_recognition.git
cd Face_recognition
pip install -r requirements.txt
```

### 2. Run:
1. Start Qdrant VectorDB:
```bash
docker-compose -f docker-compose.yml --env-file .env up -d qdrant
```
2. Run Gradio app:
```bash
python3 app/main.py
```
