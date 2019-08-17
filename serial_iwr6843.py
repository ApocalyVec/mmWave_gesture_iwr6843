import serial


def serialConfig(configFileName, dataPortName, userPortName):
    # Open the serial ports for the configuration and the data ports
    try:
        cliPort = serial.Serial(userPortName, 115200)
        dataPort = serial.Serial(dataPortName, 921600)
    except serial.SerialException as se:
        print('Serial Port Occupied, error = ')
        print(str(se))
        return

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        cliPort.write((i + '\n').encode())
        print(i)

    return cliPort, dataPort
