import mysql.connector
from mysql.connector.constants import ClientFlag
config = {
    'user': 'manasi',
    'password': 'A1_34_intern',
    'host': '193.123.68.190',
    'client_flags': [ClientFlag.SSL],
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