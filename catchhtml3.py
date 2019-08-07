import urllib.request as ur
import requests
from bs4 import BeautifulSoup
import csv
url="https://www.hltv.org/ranking/teams/2019/may/20"
response=requests.get(url)
html_doc=response.text
soup=BeautifulSoup(html_doc,"lxml")
team=soup.find_all('div',{'class':"relative"})
data=[]

for i in range(0,len(team)):
	team_data=[]
	team_info=team[i].find_all('span')
	for j in range(0,7):
		team_data.extend([team_info[j].text])
	data.append(team_data)
with open('C:/Users/user/Downloads/pythonCode/CSGO.csv','w',newline='') as f:
	writer=csv.writer(f)
	for i in range(0,len(data)):
		writer.writerow(data[i])