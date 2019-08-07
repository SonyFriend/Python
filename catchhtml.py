import urllib.request as ur
import requests
from bs4 import BeautifulSoup
url = "https://tw.yahoo.com/"
response = requests.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml") # 指定 lxml 作為解析器
stories=soup.find_all('a',class_='story-title')
print(stories)
#for s in stories:
#	print("標題:" +s.text)
#	print("網址:" +s.get('href'))