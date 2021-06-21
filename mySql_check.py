import mysql.connector
from mysql.connector.constants import ClientFlag
config = {
    'user': 'root',
    'password': 'A1_34_intern',
    'host': '193.123.68.57',
    'database' : 'books',
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': 'C:\project_Downloads\ca.pem',
    'ssl_cert': 'C:\project_Downloads\client-cert.pem',
    'ssl_key': 'C:\project_Downloads\client-key.pem',
}
try:
    cnx = mysql.connector.connect(**config)
    #cnx = mysql.connector.connect(user = 'root', password = 'A1_34_intern', host = '193.123.68.57',
    #client_flags = [ClientFlag.SSL],
    #ssl_ca ='C:\project_Downloads\ca.pem',
    #ssl_cert = 'C:\project_Downloads\client-cert.pem',
    #ssl_key = 'C:\project_Downloads\client-key.pem')
    cur = cnx.cursor(buffered=True)
    #cur.execute("SELECT count() FROM DUAL")
    #cur.execute("SHOW STATUS LIKE 'Ssl_cipher'")
    cur.execute("SELECT * FROM authors")
    print(cur.fetchone())
    cur.close()
    cnx.close()
except mysql.connector.Error as e:
    print("error occured:",e)