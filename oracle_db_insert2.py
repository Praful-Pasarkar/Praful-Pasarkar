import cx_Oracle

class oracle_db_insert2:
    'Insert user inputted data in table in oracle database'

    def __init__(self):
        'Do something'
        print("Class: oracle_db_insert2")

    # send password also
    def create_connection(username):
        'Code for creating connection to oracle db'
        connection = cx_Oracle.connect(user=username, password="A1_34_intern", dsn="almltrdb_low", encoding="UTF-8")
        cursor = connection.cursor()
        return connection, cursor

    def receive_my_input(cursor):
        'To recieve the the input from outside class'
        id1 = input("ID:\n")
        name1 = input('Name:\n')
        age1 = input('Age:\n')
        add1 = input('Address\n')
        sal1 = input('Salary\n')
        return id1, name1, age1, add1, sal1

    def insert_into_table(cursor, id1, name1,age1, add1, sal1):
        'The is code to insert data'
        mysql3 = f'INSERT INTO CUSTOMERS (ID,NAME,AGE,ADDRESS,SALARY) VALUES (:id1, :name1 , :age1, :add1, :sal1 )'
        cursor.execute(mysql3, [id1, name1, age1, add1, sal1])

    def table_update(cursor, id1, name1, age1, add1, sal1):
        mysql4 = "UPDATE CUSTOMERS SET ADDRESS = 'VENUS' WHERE ID = 231"
        cursor.execute(mysql4)

    def print_my_result(connection, cursor):
        'Pretty-fy my result'
        mysql1 = """select * from CUSTOMERS order by ID"""
        print("\n##########################")
        print("Result:")
        for result in cursor.execute(mysql1):
            print(result)
        print("##########################\n")
        print()
        connection.commit()
        connection.close()

#oracle_db_insert2('Send the values')
#oracle_db_insert2('Get a handle and call each def seperately to do the work')
username = input("Enter user: praful/mihir/manasi\n")
connection, cursor = oracle_db_insert2.create_connection(username)

id1, name1, age1, add1, sal1 = oracle_db_insert2.receive_my_input(cursor)

oracle_db_insert2.insert_into_table(cursor,id1, name1, age1, add1, sal1)
#oracle_db_insert2.table_update(cursor,id1, name1, age1, add1, sal1)
oracle_db_insert2.print_my_result(connection, cursor)