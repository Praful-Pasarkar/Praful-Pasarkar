from configparser import ConfigParser

file = 'config.ini' # The config file
config = ConfigParser()
config.read(file)

# Printing config files sections
print(config.sections())
print(list(config['user']))
print(config['user']['password'])

# Something to update the config file with
#config.add_section('Directory')
#config.set('Directory', 'Path', 'C:\\Users\\Mihir Abhyankar\\PycharmProjects\\Praful-Pasarkar')

#with open(file, 'w') as configfile:
#config.write(configfile)