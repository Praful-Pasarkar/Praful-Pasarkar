import cx_Oracle

print("test")
connection = cx_Oracle.connect('praful', 'A1_34_intern','almltrdb_low')
mysql1 = """select * from STUDENTS order by ID"""
mysql2 = """INSERT INTO STUDENTS (ID,NAME,AGE,ADDRESS) VALUES (1, 'John', 18, 'US' )"""
#mysql1= """select sysdate from dual"""
#mysql1 = """ select count(*) from dual"""

mysql3 = """select * from TEACHERS order by ID"""
mysql4 = """INSERT INTO CUSTOMERS (ID, NAME, AGE,ADDRESS) VALUES (30, 'Emma', 30, 'UK')"""

mysql5 = """select * from SCHOOLS order by NAME"""
mysql6 = """INSERT INTO SCHOOLS (EMPLOYEES, NAME,ADDRESS) VALUES (76, 'Charter school', 'US')"""

mysql7 = """select * from LOCATION order by POSTAL_CODE"""
mysql8 = """INSERT INTO LOCATION (POSTAL_CODE, NAME,POPULATION) VALUES (8437, 'Brooklyn', 850000)"""

cursor = connection.cursor()

#for result in cursor.execute(mysql1):
 #   print(result)
#print()

cursor.execute(mysql2)
for result in cursor.execute(mysql1):
    print(result)
print()

cursor.execute(mysql4)
for result in cursor.execute(mysql3):
    print(result)
print()

cursor.execute(mysql6)
for result in cursor.execute(mysql5):
    print(result)
print()

cursor.execute(mysql8)
for result in cursor.execute(mysql7):
    print(result)
print()
