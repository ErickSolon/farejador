import yolov5
import numpy
import os
import requests
from bs4 import BeautifulSoup
from PIL import Image

# Software feito pra que baixe imagens em index of. inurl:"index of"
URL = "https://pida.com.br/fotos/pida_noticias/2468/mg/"
extensao = 'jpg'
pasta_imagens = 'imagens'
procurando = 'person'

if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)
else:
    arquivos = os.listdir(pasta_imagens)
    for arquivo in arquivos:
        os.remove(f'{pasta_imagens}/{arquivo}')

imagem = requests.get(url=URL)
bs4 = BeautifulSoup(imagem.text, 'html.parser')
for imagens in bs4.find_all('a'):
    nome_arquivo = imagens.get('href')
    if nome_arquivo and nome_arquivo.endswith(f'.{extensao}'):
        img_url = URL + nome_arquivo
        img_resposta = requests.get(img_url)
        if img_resposta.status_code == 200:
            with open(f'{pasta_imagens}/{nome_arquivo}', 'wb') as arquivo:
                print(f"[+] {nome_arquivo}")
                arquivo.write(img_resposta.content)

model = yolov5.load('yolov5s.pt')
arquivos = os.listdir(pasta_imagens)

for imagem in arquivos:
    if imagem.endswith(f".{extensao}"):
        imagem = Image.open(f'{pasta_imagens}/{imagem}')
        imagem = numpy.array(imagem)
        resultado = model(imagem)

        for det in resultado.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, classe = det
            if conf > 0.5:
                nome = model.names[int(classe)]

                if nome == procurando:
                    resultado.show()
