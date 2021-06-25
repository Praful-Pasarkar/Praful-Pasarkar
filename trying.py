import cx_Oracle

print("test")
conn = cx_Oracle.connect(user='admin', password="A1_34_intern", dsn="devtrxmandb_low", encoding="UTF-8")
cursor = conn.cursor()
mysql1 = '''SELECT SYSDATE FROM DUAL'''
result = cursor.execute(mysql1)
print(result)