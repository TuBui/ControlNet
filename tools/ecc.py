import bchlib
import numpy as np 
from typing import List, Tuple
import random 


class RSC(object):
    def __init__(self, data_bytes=16, ecc_bytes=4, verbose=False):
        from reedsolo import RSCodec
        self.rs = RSCodec(ecc_bytes)
        if verbose:
            print(f'Reed-Solomon ECC len: {ecc_bytes*8} bits')
        self.dlen = data_bytes * 8  # data length in bits
        self.ecc_len = ecc_bytes * 8  # ecc length in bits
    
    def get_total_len(self):
        return self.dlen + self.ecc_len

    def encode_text(self, text: List[str]):
        return np.array([self._encode_text(t) for t in text])
    
    def _encode_text(self, text: str):
        text = text + ' ' * (self.dlen // 8 - len(text))
        out = self.rs.encode(text.encode('utf-8'))  # bytearray
        out = ''.join(format(x, '08b') for x in out)  # bit string
        out = np.array([int(x) for x in out], dtype=np.float32)
        return out
    
    def decode_text(self, data: np.array):
        assert len(data.shape)==2
        return [self._decode_text(d) for d in data]

    def _decode_text(self, data: np.array):
        assert len(data.shape)==1
        data = ''.join([str(int(bit)) for bit in data])
        data = bytes(int(data[i: i + 8], 2) for i in range(0, len(data), 8))
        data = bytearray(data)
        try:
            data = self.rs.decode(data)[0]
            data = data.decode('utf-8').strip()
        except:
            print('Error: Decode failed')
            data = get_random_unicode(self.get_total_len()//8)
        
        return data

def get_random_unicode(length):
    # Update this to include code point ranges to be sampled
    include_ranges = [
        ( 0x0021, 0x0021 ),
        ( 0x0023, 0x0026 ),
        ( 0x0028, 0x007E ),
        ( 0x00A1, 0x00AC ),
        ( 0x00AE, 0x00FF ),
        ( 0x0100, 0x017F ),
        ( 0x0180, 0x024F ),
        ( 0x2C60, 0x2C7F ),
        ( 0x16A0, 0x16F0 ),
        ( 0x0370, 0x0377 ),
        ( 0x037A, 0x037E ),
        ( 0x0384, 0x038A ),
        ( 0x038C, 0x038C ),
    ]
    alphabet = [
        chr(code_point) for current_range in include_ranges
            for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return ''.join(random.choice(alphabet) for i in range(length))

class ECC(object):
    def __init__(self, BCH_POLYNOMIAL = 137, BCH_BITS = 5):
        self.bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    
    def get_total_len(self):
        return 100

    def _encode(self, x):
        # x: 56 bits, {0, 1}, np.array
        # return: 100 bits, {0, 1}, np.array
        dlen = len(x)
        data_str = ''.join(str(x) for x in x.astype(int))
        packet = bytes(int(data_str[i: i + 8], 2) for i in range(0, dlen, 8))
        packet = bytearray(packet)
        ecc = self.bch.encode(packet)
        packet = packet + ecc  # 96 bits
        packet = ''.join(format(x, '08b') for x in packet)
        packet = [int(x) for x in packet]
        packet.extend([0, 0, 0, 0])
        packet = np.array(packet, dtype=np.float32)  # 100
        return packet

    def _decode(self, x):
        # x: 100 bits, {0, 1}, np.array
        # return: 56 bits, {0, 1}, np.array
        packet_binary = "".join([str(int(bit)) for bit in x])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bitflips = self.bch.decode_inplace(data, ecc)
        if bitflips == -1:  # error, return random data
            data = np.random.binomial(1, .5, 56) 
        else:
            data = ''.join(format(x, '08b') for x in data)
            data = np.array([int(x) for x in data], dtype=np.float32)
        return data  # 56 bits
    
    def _generate(self):
        dlen = 56
        data= np.random.binomial(1, .5, dlen)
        packet = self._encode(data)
        return packet, data

    def generate(self, nsamples=1):
        # generate random 56 bit secret
        data = [self._generate() for _ in range(nsamples)]
        data = (np.array([d[0] for d in data]), np.array([d[1] for d in data]))
        return data  # data with ecc, data org
    
    def _to_text(self, data):
        # data:  {0, 1}, np.array
        # return: str
        data = ''.join([str(int(bit)) for bit in data])
        all_bytes = [ data[i: i+8] for i in range(0, len(data), 8) ]
        text = ''.join([chr(int(byte, 2)) for byte in all_bytes])
        return text.strip()
    
    def _to_binary(self, s):
        if isinstance(s, str):
            out = ''.join([ format(ord(i), "08b") for i in s ])
        elif isinstance(s, bytes):
            out = ''.join([ format(i, "08b") for i in s ])
        elif isinstance(s, np.ndarray) and s.dtype is np.dtype(bool):
            out = ''.join([chr(int(i)) for i in s])
        elif isinstance(s, int) or isinstance(s, np.uint8):
            out = format(s, "08b")
        elif isinstance(s, np.ndarray):
            out = [ format(i, "08b") for i in s ]
        else:
            raise TypeError("Type not supported.")

        return np.array([float(i) for i in out], dtype=np.float32)

    def _encode_text(self, s):
        s = s + ' '*(7-len(s))  # 7 chars
        s = self._to_binary(s)  # 56 bits
        packet = self._encode(s)  # 100 bits
        return packet, s

    def encode_text(self, secret_list, return_pre_ecc=False):
        """encode secret with BCH ECC.
        Input: secret (list of strings)
        Output: secret (np array) with shape (B, 100) type float23, val {0,1}"""
        assert np.all(np.array([len(s) for s in secret_list]) <= 7), 'Error! all strings must be less than 7 characters'
        secret_list = [self._encode_text(s) for s in secret_list]
        ecc = np.array([s[0] for s in secret_list], dtype=np.float32)
        if return_pre_ecc:
            return ecc, np.array([s[1] for s in secret_list], dtype=np.float32)
        return ecc
    
    def decode_text(self, data):
        """Decode secret with BCH ECC and convert to string.
        Input: secret (torch.tensor) with shape (B, 100) type bool
        Output: secret (B, 56)"""
        data = self.decode(data)
        data = [self._to_text(d) for d in data]
        return data

    def decode(self, data):
        """Decode secret with BCH ECC and convert to string.
        Input: secret (torch.tensor) with shape (B, 100) type bool
        Output: secret (B, 56)"""
        data = data[:, :96]
        data = [self._decode(d) for d in data]
        return np.array(data)


if __name__ == '__main__':
    ecc = ECC()
    batch_size = 10 
    secret_ecc, secret_org = ecc.generate(batch_size)  # 10x100 ecc secret, 10x56 org secret
    # modify secret_ecc
    secret_pred = secret_ecc.copy()
    secret_pred[:,3:6] = 1 - secret_pred[:,3:6]
    # pass secret_ecc to model and get predicted as secret_pred
    secret_pred_org = ecc.decode(secret_pred)  # 10x56
    bit_ecc = np.all(secret_pred_org == secret_org, axis=1)  # 10
    print(bit_ecc)