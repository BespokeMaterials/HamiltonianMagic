import numpy as np
import chardet

def detect_encoding(bit_string):
    with open(file_path, 'rb') as file:
        data = file.read()
        result = chardet.detect(data)
    return result['encoding']

def read_bit_string_file(file_path):
    with open(file_path, 'rb') as file:
        bit_string = file.read()
    return bit_string


# Example usage
file_path="/Users/voicutomut/Desktop/MatrixCompression/bras.bin"
bitstring=read_bit_string_file(file_path)
encoding = chardet.detect(bitstring)
print("Detected encoding:", encoding)


# Reading file data with read() method
data = bitstring.decode(encoding)


# Knowing the Type of our data
print(type(data))

# Printing our byte sequenced data
print(data)

print(len(data))


x = np.reshape(data, (207*(200+256), 2096))

print(x)

