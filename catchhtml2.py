import urllib.request as ur
import requests
from bs4 import BeautifulSoup
import csv
url = "http://www.basketball-reference.com/boxscores/201611230BRK.html"
response = requests.get(url) # 用 requests 的 get 方法把網頁抓下來
html_doc = response.text # text 屬性就是 html 檔案
soup = BeautifulSoup(response.text, "lxml")
#score=soup.find_all('div',class_="sib-player-stats-paginate")
bos_basic=soup.find_all('div',{'id':'all_box_bos_basic'})[0].find_all('tbody')
NBA_data=[]
play_stat_row=bos_basic[0].find_all('tr')
Play_stat_col=bos_basic[0].find_all('tr')[0].find_all('td')

NBA_team_data=[]
for i in range(0,(len(play_stat_row)-1)):
	NBA_player_data=[]
	NBA_player_stat=bos_basic[0].find_all('tr')[i].find_all('td')
	NBA_play_name=bos_basic[0].find_all('tr')[i].find_all('th')
	for j in range(0,len(Play_stat_col)):
		if i==5:
			pass
		else:
			NBA_player_data.append(NBA_player_stat[j].text)
	NBA_player_data.insert(0,NBA_play_name[0].text)
	NBA_team_data.extend([NBA_player_data])

#with open('C:/Users/user/Downloads/pythonCode/NBA.csv','w',newline='') as f:
#	writer=csv.writer(f)
#	for i in range(0,len(NBA_team_data)):
#		writer.writerow(NBA_team_data[i])