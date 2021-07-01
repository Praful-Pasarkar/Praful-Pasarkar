import mysql.connector
from mysql.connector.constants import ClientFlag

from configparser import ConfigParser

file = 'config.ini' # The config file
config = ConfigParser()
config.read(file)
name = config['user_root']['name']
pswd = config['user_root']['password']
host = config['Server']['ip_mihir']
file1 = config['Keys']['ssl_ca']
file2 = config['Keys']['ssl_cert']
file3 = config['Keys']['ssl_key']

print(name)
print(pswd)
config = {
    'user': name,
    'password': pswd,  # Take it as an input
    'host': host,     # Take it as an input
    'database' : 'books',
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': file1,
    'ssl_cert': file2,
    'ssl_key': file3,
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