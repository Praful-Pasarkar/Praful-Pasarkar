import cx_Oracle

print("test")
connection = cx_Oracle.connect('praful', 'A1_34_intern','almltrdb_low')
mysql1 = """select * from EMPLOYEE order by ID"""
mysql1= """select sysdate from dual"""

mysql1 = """ select count(*) from dual"""

print("Get all rows")
cursor = connection.cursor()
for result in cursor.execute(mysql1):
    print(result)
print()
