import serial_iwr6843

configFileName = 'iwr6843_config_default.cfg'
dataPortName = 'COM9'
userPortName = 'COM8'

# open the serial port to the radar
userPort, dataPort = serial_iwr6843.serialConfig(configFileName, dataPortName=dataPortName, userPortName=userPortName)
# configParameters = parseConfigFile(configFileName)

while 1:
    pass