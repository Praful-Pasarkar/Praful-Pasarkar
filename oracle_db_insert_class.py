import cx_Oracle

class oracle_insert:
    print('Enter CUSTOMER Data values')
    id1 = input('ID:\n')
    name1 = input('Name:\n')
    age1 = input('Age:\n')
    add1 = input('Address\n')
    sal1 = input('Salary\n')
    connection = cx_Oracle.connect(user="praful", password="A1_34_intern",dsn = "almltrdb_low", encoding="UTF-8")
    mysql1 = """select * from CUSTOMERS order by ID"""
    #mysql2 = """INSERT INTO CUSTOMERS (ID,NAME,AGE,ADDRESS,SALARY) VALUES (1,'M', 13, 'f', 100 )"""
    cursor = connection.cursor()
    mysql3 = f'INSERT INTO CUSTOMERS (ID,NAME,AGE,ADDRESS,SALARY) VALUES (:id1, :name1 , :age1, :add1, :sal1 )'
    cursor.execute(mysql3, [id1, name1, age1, add1, sal1])
    for result in cursor.execute(mysql1):
        print(result)
    print()
    connection.close()

