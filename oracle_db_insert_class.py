import cx_Oracle

class oracle_insert:
    print('Enter CUSTOMER Data values')
    id = input('ID:\n')
    name = input('Name:\n')
    age = input('Age:\n')
    add = input('Address\n')
    sal = input('Salary\n')
    connection = cx_Oracle.connect(user="praful", password="A1_34_intern",dsn = "almltrdb_low", encoding="UTF-8")
    mysql1 = """select * from CUSTOMERS order by ID"""
    mysql2 = """INSERT INTO CUSTOMERS (ID,NAME,AGE,ADDRESS,SALARY) VALUES (?, ?, ?, ?, ? )"""
    data = (id, name, age, add, sal);
    cursor = connection.cursor()
    cursor.execute(mysql2, data)
    for result in cursor.execute(mysql1):
        print(result)
    print()
