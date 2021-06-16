import mysql.connector
from mysql.connector.constants import ClientFlag
config = {
    'user': 'root',
    'password': 'A1_34_intern',
    'host': '193.123.69.60',
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': 'C:\project_Downloads\ca.pem',
    'ssl_cert': 'C:\project_Downloads\client-cert.pem',
    'ssl_key': 'C:\project_Downloads\client-key.pem',
}
try:
    cnx = mysql.connector.connect(**config)
    cur = cnx.cursor(buffered=True)
    cur.execute("SELECT count() FROM DUAL")
    print(cur.fetchone())
    cur.close()
    cnx.close()
except mysql.connector.Error as e:
    print("error occured:",e)