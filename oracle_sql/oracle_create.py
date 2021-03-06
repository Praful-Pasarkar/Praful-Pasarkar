import cx_Oracle

class oracle_create:

    def __init__(self):
        #'Do something'
        print("Class: oracle_create")

    # Mihir: almltrdb_low, Manasi: devtrxmandb_low
    def create_connection(username):
        #'Code for creating connection to oracle db'
        connection = cx_Oracle.connect(user=username, password="A1_34_intern", dsn="devtrxmandb_low", encoding="UTF-8")
        cursor = connection.cursor()
        return connection, cursor

    def create_table(cursor, name):
        # Create table
        cre = """CREATE TABLE {tab} (ID NUMBER GENERATED BY DEFAULT ON NULL AS IDENTITY, 
        NAME VARCHAR (20) NOT NULL, AGE  INT              NOT NULL,
        ADDRESS  CHAR (25) ,SALARY   DECIMAL (18, 2))
        FOREIGN KEY(ADDRESS) REFERENCES suppliers(group_id)""".format(tab=name)
        cursor.execute(cre)

    def input(cursor):
        # Recieve input
        name = input("Table Name\n")
        name1 = input('Name:\n')
        age1 = input('Age:\n')
        add1 = input('Address\n')
        sal1 = input('Salary\n')
        return name, name1, age1, add1, sal1

    def insert_table(cursor, name1,age1, add1, sal1):
        mysql3 = f'INSERT INTO CUSTOMERS (NAME,AGE,ADDRESS,SALARY) VALUES (:name1 , :age1, :add1, :sal1 )'
        cursor.execute(mysql3, [name1, age1, add1, sal1])

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

username = input("Username\n")
connection, cursor = oracle_create.create_connection(username)
name, name1, age1, add1, sal1 = oracle_create.input(cursor)
#oracle_create.create_table(cursor, name)
oracle_create.insert_table(cursor, name1, age1, add1, sal1)
oracle_create.print_my_result(connection, cursor)



#FOREIGN KEY(ADDRESS) REFERENCES suppliers(group_id)

