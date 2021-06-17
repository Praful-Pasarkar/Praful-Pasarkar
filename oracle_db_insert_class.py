import cx_Oracle

class oracle_insert:
    print("test")
    connection = cx_Oracle.connect('praful', 'A1_34_intern','almltrdb_low')
    mysql1 = """select * from CUSTOMERS order by ID"""
    mysql2 = """INSERT INTO CUSTOMERS (ID,NAME,AGE,ADDRESS,SALARY) VALUES (10, 'K', 21, 'MQ', 400 )"""
    print("Get all rows")
    cursor = connection.cursor()
    cursor.execute(mysql2)
    for result in cursor.execute(mysql1):
        print(result)
    print()
