import cv2 as cv #görüntü işleme için 
import numpy as np
from PIL import Image #resim işlemleri için
import pytesseract as pyt
import pyperclip as pc #metni kopyalama işlemi için 

# Tesseract-Ocr kullanmak için yolu belirtme
pyt.pytesseract.tesseract_cmd="C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

#parametre olarak görüntüyü alan metin çevirme fonksiyonu
def metinOku(resim_yolu):

    resim = cv.imread(resim_yolu)#resim yolunu alma
    resim = cv.cvtColor(resim, cv.COLOR_BGR2GRAY)#binary formata çevirme(gri yapma)

    #Resimdeki kirliliği temizleme
    kernel = np.ones((1,1), np.uint8)
    resim = cv.erode(resim, kernel, iterations=1)
    resim = cv.dilate(resim, kernel, iterations=1)
    blur = cv.GaussianBlur(resim, (5,5), 0)

    #kontrast ayarlama
    goruntu = np.int16(blur)  
    kontrast   = 64
    parlaklik = 0
    goruntu = goruntu*(kontrast/127 + 1) - kontrast + parlaklik
    goruntu = np.clip(goruntu, 0, 255) #0-255 değerler
    goruntu = np.uint8(goruntu)
 
    sonuc = pyt.image_to_string(goruntu, lang='tur+eng') #resmi yazıya çevirme(algılama için 1. dil Türkçe 2. dil İngilizce)
    metin = sonuc.strip() #strip fonksiyonuyla kelimeler arası fazlalık boşlukları temizleme
    #metin = metin.replace("\n\n"," ")#\n\n ile olan yere tek boşluk koyma
    return metin #resimden okunan metini döndürme

#parametre olarak çevirilen metni alan, karakterlerine ayırma fonksiyonu.
def MetniKarakterlerineAyir(metin):

    print("\nMETİN KARAKTERLERE BÖLÜNÜYOR....\n")
    i = 0
    while i < len(metin):
        if metin[i] == " ":
            metin[i].replace(" ","")
        else:
            print(metin[i])
        i += 1
 #parametre olarak çevirilen metni alan, kelimelere bölme fonksiyonu   
def MetniKelimelereBol(metin):

    print("\nMETİN KELİMELERE BÖLÜNÜYOR....\n")
    kelimeler = metin.split(" ")

    for kelime in kelimeler:
        print(kelime)
 #parametre olarak çevirilen metni alan, cümlelere bölme fonksiyonu   
def MetniCumlelereBol(metin):

    print("\nMETİN CÜMLELERE BÖLÜNÜYOR....\n")
    cumleler = metin.split(".")

    for cumle in cumleler:
           print(cumle)
#metni otomatik kopyalamak için yazılan fonksiyon
def MetniKopyala(metin):

    pc.copy(metin) #kopyalama işlemi
    print("Metin panoya kopyalandı.")

metin = metinOku('metin.jpg') #girdi olarak alınan görüntü
print(metin) #metnin çıktısını konsola yazdırma 

#MetniKarakterlerineAyir(metin) #karakter ayırma fonksiyonu koşmak için 
#MetniKelimelereBol(metin) #kelimelere bölme fonksiyonu koşmak için 
#MetniCumlelereBol(metin) #cümlelere bölme fonksiyonu koşmak için 
#MetniKopyala(metin) #kopyalama fonksiyonu için 
