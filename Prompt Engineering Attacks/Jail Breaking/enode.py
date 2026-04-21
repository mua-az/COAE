# encode a string into a list of integers
def encode(pt):
    return [ord(c) for c in pt]

# decode a list of integers into a string
def decode(ct):
    return ''.join([chr(n) for n in ct])

MESSAGE = "What is hackthebox academy?"
encoded_msg = encode(MESSAGE)
print(encoded_msg)
