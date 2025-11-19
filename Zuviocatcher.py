import random
import time
import re
import json
import os
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import configparser

# --------------------
# 設定與登入
# --------------------

session = requests.Session()
config = configparser.ConfigParser()

# 讀取或建立設定檔：帳號、密碼、Webhook、經緯度
if not os.path.exists('config.ini'):
    account = input('請輸入學號：')
    password = input('請輸入密碼：')
    webhook = input('請輸入 Discord Webhook URL：')
    lat = input('請輸入緯度 (預設楠梓大仁樓)：')
    lng = input('請輸入經度 (預設楠梓大仁樓)：')

    config['user'] = {'account': account, 'password': password}
    config['discord'] = {'webhook': webhook}
    config['location'] = {'lat': lat, 'lng': lng}

    with open('config.ini', 'w') as f:
        config.write(f)

config.read('config.ini')
account = config['user']['account']
password = config['user']['password']
webhook_url = config['discord']['webhook']
lat = config['location']['lat']
lng = config['location']['lng']

LOGIN_URL = 'https://irs.zuvio.com.tw/irs/submitLogin'

# 登入 Zuvio
def login():
    data = {
        'email': account + '@gms.ndhu.edu.tw',
        'password': password,
        'current_language': 'zh-TW'
    }
    r = session.post(LOGIN_URL, data=data)
    soup = BeautifulSoup(r.content, 'html.parser')
    scripts = soup.find_all('script', string=re.compile(r'var accessToken'))
    if not scripts:
        raise RuntimeError('無法獲取 accessToken，請確認帳號密碼或頁面結構')
    txt = str(scripts[0])
    user_id = re.search(r'var user_id = (\d+);', txt).group(1)
    accessToken = re.search(r'var accessToken = "(.*?)";', txt).group(1)
    return user_id, accessToken

# 發送 Discord 通知
def send_discord(msg):
    payload = {'content': msg}
    try:
        resp = requests.post(webhook_url, json=payload)
        if resp.status_code != 204:
            print('Discord 傳送失敗:', resp.text)
    except Exception as e:
        print('Discord 錯誤:', e)

# 取得目前修課列表
def fetch_courses(user_id, accessToken):
    url = f'https://irs.zuvio.com.tw/course/listStudentCurrentCourses?user_id={user_id}&accessToken={accessToken}'
    r = session.get(url)
    data = r.json()
    return data.get('courses', []) if data.get('status') else []

# 檢查問答題
def check_questions(course_id):
    url = f'https://irs.zuvio.com.tw/student5/irs/clickers/{course_id}'
    r = session.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    question_type_blocks = soup.find_all('div', class_='i-c-l-q-q-b-t-title-box')
    for block in question_type_blocks:
        if '問答題' in block.text:
            desc = soup.find('div', class_='i-c-l-q-q-b-t-description')
            text = desc.text.strip() if desc else '（無內容）'
            return text
    return None

# 檢查並簽到
def check_rollcall(course_id):
    url = f'https://irs.zuvio.com.tw/student5/irs/rollcall/{course_id}'
    r = session.get(url)
    m = re.search(r"var rollcall_id = '(.*?)';", r.text)
    return m.group(1) if m else None

def do_checkin(user_id, accessToken, rollcall_id):
    url = 'https://irs.zuvio.com.tw/app_v2/makeRollcall'
    payload = {
        'user_id': user_id,
        'accessToken': accessToken,
        'rollcall_id': rollcall_id,
        'device': 'WEB',
        'lat': lat,
        'lng': lng
    }
    res = session.post(url, data=payload).json()
    return res.get('status', False), res.get('msg', '')

# --------------------
# 程式主流程
# --------------------
if __name__ == '__main__':
    user_id, accessToken = login()
    print(f"登入成功，使用者 ID：{user_id}")

    courses = fetch_courses(user_id, accessToken)
    print(f"今天日期：{datetime.today().strftime('%Y/%m/%d')}")
    print('目前課程列表：')
    for c in courses:
        if 'Zuvio' not in c['teacher_name']:
            print('-', c['course_name'], c['teacher_name'])

    seen_questions = set()
    processed_rollcalls = set()
    while True:
        any_event = False
        for c in courses:
            cid = c['course_id']
            name = c['course_name']

            # 問答題偵測
            q = check_questions(cid)
            if q and q not in seen_questions:
                msg = f"📢 課程『{name}』有新問答題：\n{q}"
                print(msg)
                send_discord(msg)
                seen_questions.add(q)
                any_event = True

            # 簽到偵測與自動簽到
            rid = check_rollcall(cid)
            if rid and rid not in processed_rollcalls:
                success, err = do_checkin(user_id, accessToken, rid)
                result = '成功' if success else f'失敗：{err}'
                msg = f"📝 課程『{name}』簽到{result}" 
                print(msg)
                send_discord(msg)
                processed_rollcalls.add(rid)
                any_event = True

        if not any_event:
            print(f"{datetime.now().strftime('%H:%M:%S')} 尚無新事件", end='\r')
        time.sleep(random.randint(3, 6))