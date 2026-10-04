import sqlite3

def create():
    conn = sqlite3.connect("DataBase.db")
    cursor = conn.cursor()
    cursor.execute("create table if not exists user (Id integer primary key AUTOINCREMENT,file_name text,date text,prediction text,confidence real)")
    conn.commit()
    conn.close()

def fill(file_name,date,prediction,confidence):
    conn = sqlite3.connect("DataBase.db")
    cursor = conn.cursor()
    cursor.execute("insert into user(file_name,date,prediction,confidence) values(?,?,?,?)",(file_name,date,prediction,confidence))
    conn.commit()
    conn.close()

def display():
    conn = sqlite3.connect("DataBase.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user")
    data = cursor.fetchall()
    conn.close()
    return data