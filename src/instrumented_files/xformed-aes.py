import sys
from instrument_lib import *
from loop import loop
import numpy as np
print(1, 3)
Sbox_0 = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 
    171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156,
    164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 
    113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 
    235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 
    41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57,
    74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 
    80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 
    33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126,
    61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184,
    20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98,
    145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244,
    234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 
    116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 
    87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155,
    30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 
    65, 153, 45, 15, 176, 84, 187, 22)
write_instrument_read(Sbox_0, 'Sbox_0')
if type(Sbox_0) == np.ndarray:
    print('malloc', sys.getsizeof(Sbox_0), 'Sbox_0', Sbox_0.shape)
elif type(Sbox_0) == list:
    dims = []
    tmp = Sbox_0
    while type(tmp) == list:
        dims.append(len(tmp))
        if len(tmp) > 0:
            tmp = tmp[0]
        else:
            tmp = None
    print('malloc', sys.getsizeof(Sbox_0), 'Sbox_0', dims)
elif type(Sbox_0) == tuple:
    print('malloc', sys.getsizeof(Sbox_0), 'Sbox_0', [len(Sbox_0)])
else:
    print('malloc', sys.getsizeof(Sbox_0), 'Sbox_0')
print(1, 20)
InvSbox_0 = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243,
    215, 251, 124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 
    222, 233, 203, 84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66,
    250, 195, 78, 8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109,
    139, 209, 37, 114, 248, 246, 100, 134, 104, 152, 22, 212, 164, 92, 204,
    93, 101, 182, 146, 108, 112, 72, 80, 253, 237, 185, 218, 94, 21, 70, 87,
    167, 141, 157, 132, 144, 216, 171, 0, 140, 188, 211, 10, 247, 228, 88, 
    5, 184, 179, 69, 6, 208, 44, 30, 143, 202, 63, 15, 2, 193, 175, 189, 3,
    1, 19, 138, 107, 58, 145, 17, 65, 79, 103, 220, 234, 151, 242, 207, 206,
    240, 180, 230, 115, 150, 172, 116, 34, 231, 173, 53, 133, 226, 249, 55,
    232, 28, 117, 223, 110, 71, 241, 26, 113, 29, 41, 197, 137, 111, 183, 
    98, 14, 170, 24, 190, 27, 252, 86, 62, 75, 198, 210, 121, 32, 154, 219,
    192, 254, 120, 205, 90, 244, 31, 221, 168, 51, 136, 7, 199, 49, 177, 18,
    16, 89, 39, 128, 236, 95, 96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 
    122, 159, 147, 201, 156, 239, 160, 224, 59, 77, 174, 42, 245, 176, 200,
    235, 187, 60, 131, 83, 153, 97, 23, 43, 4, 126, 186, 119, 214, 38, 225,
    105, 20, 99, 85, 33, 12, 125)
write_instrument_read(InvSbox_0, 'InvSbox_0')
if type(InvSbox_0) == np.ndarray:
    print('malloc', sys.getsizeof(InvSbox_0), 'InvSbox_0', InvSbox_0.shape)
elif type(InvSbox_0) == list:
    dims = []
    tmp = InvSbox_0
    while type(tmp) == list:
        dims.append(len(tmp))
        if len(tmp) > 0:
            tmp = tmp[0]
        else:
            tmp = None
    print('malloc', sys.getsizeof(InvSbox_0), 'InvSbox_0', dims)
elif type(InvSbox_0) == tuple:
    print('malloc', sys.getsizeof(InvSbox_0), 'InvSbox_0', [len(InvSbox_0)])
else:
    print('malloc', sys.getsizeof(InvSbox_0), 'InvSbox_0')


def xtime(a):
    print('enter scope 1')
    print(1, 39)
    print(3, 40)
    a_1 = instrument_read(a, 'a')
    write_instrument_read(a_1, 'a_1')
    if type(a_1) == np.ndarray:
        print('malloc', sys.getsizeof(a_1), 'a_1', a_1.shape)
    elif type(a_1) == list:
        dims = []
        tmp = a_1
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(a_1), 'a_1', dims)
    elif type(a_1) == tuple:
        print('malloc', sys.getsizeof(a_1), 'a_1', [len(a_1)])
    else:
        print('malloc', sys.getsizeof(a_1), 'a_1')
    print('exit scope 1')
    return (instrument_read(a_1, 'a_1') << 1 ^ 27) & 255 if instrument_read(a_1
        , 'a_1') & 128 else instrument_read(a_1, 'a_1') << 1
    print('exit scope 1')


print(1, 44)
Rcon_0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 
    47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
write_instrument_read(Rcon_0, 'Rcon_0')
if type(Rcon_0) == np.ndarray:
    print('malloc', sys.getsizeof(Rcon_0), 'Rcon_0', Rcon_0.shape)
elif type(Rcon_0) == list:
    dims = []
    tmp = Rcon_0
    while type(tmp) == list:
        dims.append(len(tmp))
        if len(tmp) > 0:
            tmp = tmp[0]
        else:
            tmp = None
    print('malloc', sys.getsizeof(Rcon_0), 'Rcon_0', dims)
elif type(Rcon_0) == tuple:
    print('malloc', sys.getsizeof(Rcon_0), 'Rcon_0', [len(Rcon_0)])
else:
    print('malloc', sys.getsizeof(Rcon_0), 'Rcon_0')


def text2matrix(text):
    print('enter scope 2')
    print(1, 48)
    print(7, 49)
    text_2 = instrument_read(text, 'text')
    write_instrument_read(text_2, 'text_2')
    if type(text_2) == np.ndarray:
        print('malloc', sys.getsizeof(text_2), 'text_2', text_2.shape)
    elif type(text_2) == list:
        dims = []
        tmp = text_2
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(text_2), 'text_2', dims)
    elif type(text_2) == tuple:
        print('malloc', sys.getsizeof(text_2), 'text_2', [len(text_2)])
    else:
        print('malloc', sys.getsizeof(text_2), 'text_2')
    print(7, 50)
    matrix_2 = []
    write_instrument_read(matrix_2, 'matrix_2')
    if type(matrix_2) == np.ndarray:
        print('malloc', sys.getsizeof(matrix_2), 'matrix_2', matrix_2.shape)
    elif type(matrix_2) == list:
        dims = []
        tmp = matrix_2
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(matrix_2), 'matrix_2', dims)
    elif type(matrix_2) == tuple:
        print('malloc', sys.getsizeof(matrix_2), 'matrix_2', [len(matrix_2)])
    else:
        print('malloc', sys.getsizeof(matrix_2), 'matrix_2')
    for i_2 in range(16):
        print(9, 52)
        byte_2 = instrument_read(text_2, 'text_2') >> 8 * (15 -
            instrument_read(i_2, 'i_2')) & 255
        write_instrument_read(byte_2, 'byte_2')
        if type(byte_2) == np.ndarray:
            print('malloc', sys.getsizeof(byte_2), 'byte_2', byte_2.shape)
        elif type(byte_2) == list:
            dims = []
            tmp = byte_2
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(byte_2), 'byte_2', dims)
        elif type(byte_2) == tuple:
            print('malloc', sys.getsizeof(byte_2), 'byte_2', [len(byte_2)])
        else:
            print('malloc', sys.getsizeof(byte_2), 'byte_2')
        if instrument_read(i_2, 'i_2') % 4 == 0:
            instrument_read(matrix_2, 'matrix_2').append([instrument_read(
                byte_2, 'byte_2')])
        else:
            instrument_read_sub(instrument_read(matrix_2, 'matrix_2'),
                'matrix_2', int(instrument_read(i_2, 'i_2') / 4), None,
                None, False).append(instrument_read(byte_2, 'byte_2'))
    print('exit scope 2')
    return instrument_read(matrix_2, 'matrix_2')
    print('exit scope 2')


def matrix2text(matrix):
    print('enter scope 3')
    print(1, 60)
    print(17, 61)
    matrix_3 = instrument_read(matrix, 'matrix')
    write_instrument_read(matrix_3, 'matrix_3')
    if type(matrix_3) == np.ndarray:
        print('malloc', sys.getsizeof(matrix_3), 'matrix_3', matrix_3.shape)
    elif type(matrix_3) == list:
        dims = []
        tmp = matrix_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(matrix_3), 'matrix_3', dims)
    elif type(matrix_3) == tuple:
        print('malloc', sys.getsizeof(matrix_3), 'matrix_3', [len(matrix_3)])
    else:
        print('malloc', sys.getsizeof(matrix_3), 'matrix_3')
    print(17, 62)
    text_3 = 0
    write_instrument_read(text_3, 'text_3')
    if type(text_3) == np.ndarray:
        print('malloc', sys.getsizeof(text_3), 'text_3', text_3.shape)
    elif type(text_3) == list:
        dims = []
        tmp = text_3
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(text_3), 'text_3', dims)
    elif type(text_3) == tuple:
        print('malloc', sys.getsizeof(text_3), 'text_3', [len(text_3)])
    else:
        print('malloc', sys.getsizeof(text_3), 'text_3')
    for i_3 in range(4):
        instrument_read(loop, 'loop').start_unroll
        for j_3 in range(4):
            print(22, 66)
            text_3 |= instrument_read_sub(instrument_read_sub(
                instrument_read(matrix_3, 'matrix_3'), 'matrix_3',
                instrument_read(i_3, 'i_3'), None, None, False),
                'matrix_3[i_3]', instrument_read(j_3, 'j_3'), None, None, False
                ) << 120 - 8 * (4 * instrument_read(i_3, 'i_3') +
                instrument_read(j_3, 'j_3'))
            write_instrument_read(text_3, 'text_3')
        instrument_read(loop, 'loop').stop_unroll
    print('exit scope 3')
    return instrument_read(text_3, 'text_3')
    print('exit scope 3')


class AES:

    def __init__(self, master_key):
        print('enter scope 4')
        print(1, 73)
        print(27, 74)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(27, 75)
        master_key_4 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key_4, 'master_key_4')
        if type(master_key_4) == np.ndarray:
            print('malloc', sys.getsizeof(master_key_4), 'master_key_4',
                master_key_4.shape)
        elif type(master_key_4) == list:
            dims = []
            tmp = master_key_4
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(master_key_4), 'master_key_4', dims)
        elif type(master_key_4) == tuple:
            print('malloc', sys.getsizeof(master_key_4), 'master_key_4', [
                len(master_key_4)])
        else:
            print('malloc', sys.getsizeof(master_key_4), 'master_key_4')
        instrument_read(self, 'self').change_key(instrument_read(
            master_key_4, 'master_key_4'))
        print('exit scope 4')

    def change_key(self, master_key):
        print('enter scope 5')
        print(1, 78)
        print(30, 79)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(30, 80)
        master_key_5 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key_5, 'master_key_5')
        if type(master_key_5) == np.ndarray:
            print('malloc', sys.getsizeof(master_key_5), 'master_key_5',
                master_key_5.shape)
        elif type(master_key_5) == list:
            dims = []
            tmp = master_key_5
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(master_key_5), 'master_key_5', dims)
        elif type(master_key_5) == tuple:
            print('malloc', sys.getsizeof(master_key_5), 'master_key_5', [
                len(master_key_5)])
        else:
            print('malloc', sys.getsizeof(master_key_5), 'master_key_5')
        print(30, 81)
        instrument_read(self, 'self').round_keys = text2matrix(instrument_read
            (master_key_5, 'master_key_5'))
        for i_5 in range(4, 4 * 11):
            instrument_read(self, 'self').round_keys.append([])
            if instrument_read(i_5, 'i_5') % 4 == 0:
                print(34, 85)
                byte_5 = instrument_read_sub(instrument_read_sub(
                    instrument_read(self, 'self').round_keys,
                    'self.round_keys', instrument_read(i_5, 'i_5') - 4,
                    None, None, False), 'self.round_keys[i_5 - 4]', 0, None,
                    None, False) ^ instrument_read_sub(instrument_read(
                    Sbox_0, 'Sbox_0'), 'Sbox_0', instrument_read_sub(
                    instrument_read_sub(instrument_read(self, 'self').
                    round_keys, 'self.round_keys', instrument_read(i_5,
                    'i_5') - 1, None, None, False),
                    'self.round_keys[i_5 - 1]', 1, None, None, False), None,
                    None, False) ^ instrument_read_sub(instrument_read(
                    Rcon_0, 'Rcon_0'), 'Rcon_0', int(instrument_read(i_5,
                    'i_5') / 4), None, None, False)
                write_instrument_read(byte_5, 'byte_5')
                if type(byte_5) == np.ndarray:
                    print('malloc', sys.getsizeof(byte_5), 'byte_5', byte_5
                        .shape)
                elif type(byte_5) == list:
                    dims = []
                    tmp = byte_5
                    while type(tmp) == list:
                        dims.append(len(tmp))
                        if len(tmp) > 0:
                            tmp = tmp[0]
                        else:
                            tmp = None
                    print('malloc', sys.getsizeof(byte_5), 'byte_5', dims)
                elif type(byte_5) == tuple:
                    print('malloc', sys.getsizeof(byte_5), 'byte_5', [len(
                        byte_5)])
                else:
                    print('malloc', sys.getsizeof(byte_5), 'byte_5')
                instrument_read_sub(instrument_read(self, 'self').
                    round_keys, 'self.round_keys', instrument_read(i_5,
                    'i_5'), None, None, False).append(instrument_read(
                    byte_5, 'byte_5'))
                for j_5 in range(1, 4):
                    print(40, 89)
                    byte_5 = instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i_5, 'i_5') - 4,
                        None, None, False), 'self.round_keys[i_5 - 4]',
                        instrument_read(j_5, 'j_5'), None, None, False
                        ) ^ instrument_read_sub(instrument_read(Sbox_0,
                        'Sbox_0'), 'Sbox_0', instrument_read_sub(
                        instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i_5,
                        'i_5') - 1, None, None, False),
                        'self.round_keys[i_5 - 1]', (instrument_read(j_5,
                        'j_5') + 1) % 4, None, None, False), None, None, False)
                    write_instrument_read(byte_5, 'byte_5')
                    if type(byte_5) == np.ndarray:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5',
                            byte_5.shape)
                    elif type(byte_5) == list:
                        dims = []
                        tmp = byte_5
                        while type(tmp) == list:
                            dims.append(len(tmp))
                            if len(tmp) > 0:
                                tmp = tmp[0]
                            else:
                                tmp = None
                        print('malloc', sys.getsizeof(byte_5), 'byte_5', dims)
                    elif type(byte_5) == tuple:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5', [
                            len(byte_5)])
                    else:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i_5,
                        'i_5'), None, None, False).append(instrument_read(
                        byte_5, 'byte_5'))
            else:
                for j_5 in range(4):
                    print(37, 94)
                    byte_5 = instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i_5, 'i_5') - 4,
                        None, None, False), 'self.round_keys[i_5 - 4]',
                        instrument_read(j_5, 'j_5'), None, None, False
                        ) ^ instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i_5, 'i_5') - 1,
                        None, None, False), 'self.round_keys[i_5 - 1]',
                        instrument_read(j_5, 'j_5'), None, None, False)
                    write_instrument_read(byte_5, 'byte_5')
                    if type(byte_5) == np.ndarray:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5',
                            byte_5.shape)
                    elif type(byte_5) == list:
                        dims = []
                        tmp = byte_5
                        while type(tmp) == list:
                            dims.append(len(tmp))
                            if len(tmp) > 0:
                                tmp = tmp[0]
                            else:
                                tmp = None
                        print('malloc', sys.getsizeof(byte_5), 'byte_5', dims)
                    elif type(byte_5) == tuple:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5', [
                            len(byte_5)])
                    else:
                        print('malloc', sys.getsizeof(byte_5), 'byte_5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i_5,
                        'i_5'), None, None, False).append(instrument_read(
                        byte_5, 'byte_5'))
        print('exit scope 5')

    def encrypt(self, plaintext):
        print('enter scope 6')
        print(1, 98)
        print(44, 99)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(44, 100)
        plaintext_6 = instrument_read(plaintext, 'plaintext')
        write_instrument_read(plaintext_6, 'plaintext_6')
        if type(plaintext_6) == np.ndarray:
            print('malloc', sys.getsizeof(plaintext_6), 'plaintext_6',
                plaintext_6.shape)
        elif type(plaintext_6) == list:
            dims = []
            tmp = plaintext_6
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(plaintext_6), 'plaintext_6', dims)
        elif type(plaintext_6) == tuple:
            print('malloc', sys.getsizeof(plaintext_6), 'plaintext_6', [len
                (plaintext_6)])
        else:
            print('malloc', sys.getsizeof(plaintext_6), 'plaintext_6')
        print(44, 101)
        instrument_read(self, 'self').plain_state = text2matrix(instrument_read
            (plaintext_6, 'plaintext_6'))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').plain_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, None, 4, True))
        for i_6 in range(1, 10):
            instrument_read(self, 'self').__round_encrypt(instrument_read(
                self, 'self').plain_state, instrument_read_sub(
                instrument_read(self, 'self').round_keys, 'self.round_keys',
                None, 4 * instrument_read(i_6, 'i_6'),
                4 * (instrument_read(i_6, 'i_6') + 1), True))
        instrument_read(self, 'self').__sub_bytes(instrument_read(self,
            'self').plain_state)
        instrument_read(self, 'self').__shift_rows(instrument_read(self,
            'self').plain_state)
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').plain_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, 40, None, True))
        print('exit scope 6')
        return matrix2text(instrument_read(self, 'self').plain_state)
        print('exit scope 6')

    def decrypt(self, ciphertext):
        print('enter scope 7')
        print(1, 111)
        print(51, 112)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(51, 113)
        ciphertext_7 = instrument_read(ciphertext, 'ciphertext')
        write_instrument_read(ciphertext_7, 'ciphertext_7')
        if type(ciphertext_7) == np.ndarray:
            print('malloc', sys.getsizeof(ciphertext_7), 'ciphertext_7',
                ciphertext_7.shape)
        elif type(ciphertext_7) == list:
            dims = []
            tmp = ciphertext_7
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(ciphertext_7), 'ciphertext_7', dims)
        elif type(ciphertext_7) == tuple:
            print('malloc', sys.getsizeof(ciphertext_7), 'ciphertext_7', [
                len(ciphertext_7)])
        else:
            print('malloc', sys.getsizeof(ciphertext_7), 'ciphertext_7')
        print(51, 114)
        instrument_read(self, 'self').cipher_state = text2matrix(
            instrument_read(ciphertext_7, 'ciphertext_7'))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').cipher_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, 40, None, True))
        instrument_read(self, 'self').__inv_shift_rows(instrument_read(self,
            'self').cipher_state)
        instrument_read(self, 'self').__inv_sub_bytes(instrument_read(self,
            'self').cipher_state)
        for i_7 in range(9, 0, -1):
            instrument_read(self, 'self').__round_decrypt(instrument_read(
                self, 'self').cipher_state, instrument_read_sub(
                instrument_read(self, 'self').round_keys, 'self.round_keys',
                None, 4 * instrument_read(i_7, 'i_7'),
                4 * (instrument_read(i_7, 'i_7') + 1), True))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').cipher_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, None, 4, True))
        print('exit scope 7')
        return matrix2text(instrument_read(self, 'self').cipher_state)
        print('exit scope 7')

    def __add_round_key(self, s, k):
        print('enter scope 8')
        print(1, 124)
        print(58, 125)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(58, 126)
        s_8 = instrument_read(s, 's')
        write_instrument_read(s_8, 's_8')
        if type(s_8) == np.ndarray:
            print('malloc', sys.getsizeof(s_8), 's_8', s_8.shape)
        elif type(s_8) == list:
            dims = []
            tmp = s_8
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_8), 's_8', dims)
        elif type(s_8) == tuple:
            print('malloc', sys.getsizeof(s_8), 's_8', [len(s_8)])
        else:
            print('malloc', sys.getsizeof(s_8), 's_8')
        print(58, 127)
        k_8 = instrument_read(k, 'k')
        write_instrument_read(k_8, 'k_8')
        if type(k_8) == np.ndarray:
            print('malloc', sys.getsizeof(k_8), 'k_8', k_8.shape)
        elif type(k_8) == list:
            dims = []
            tmp = k_8
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(k_8), 'k_8', dims)
        elif type(k_8) == tuple:
            print('malloc', sys.getsizeof(k_8), 'k_8', [len(k_8)])
        else:
            print('malloc', sys.getsizeof(k_8), 'k_8')
        for i_8 in range(4):
            for j_8 in range(4):
                print(62, 130)
                s_8[instrument_read(i_8, 'i_8')][instrument_read(j_8, 'j_8')
                    ] ^= instrument_read_sub(instrument_read_sub(
                    instrument_read(k_8, 'k_8'), 'k_8', instrument_read(i_8,
                    'i_8'), None, None, False), 'k_8[i_8]', instrument_read
                    (j_8, 'j_8'), None, None, False)
                write_instrument_read_sub(s_8[instrument_read(
                    instrument_read(i_8, 'i_8'), 'i_8')],
                    "s_8[instrument_read(i_8, 'i_8')]", instrument_read(
                    instrument_read(j_8, 'j_8'), 'j_8'), None, None, False)
        print('exit scope 8')

    def __round_encrypt(self, state_matrix, key_matrix):
        print('enter scope 9')
        print(1, 132)
        print(66, 133)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(66, 134)
        state_matrix_9 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix_9, 'state_matrix_9')
        if type(state_matrix_9) == np.ndarray:
            print('malloc', sys.getsizeof(state_matrix_9), 'state_matrix_9',
                state_matrix_9.shape)
        elif type(state_matrix_9) == list:
            dims = []
            tmp = state_matrix_9
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(state_matrix_9), 'state_matrix_9',
                dims)
        elif type(state_matrix_9) == tuple:
            print('malloc', sys.getsizeof(state_matrix_9), 'state_matrix_9',
                [len(state_matrix_9)])
        else:
            print('malloc', sys.getsizeof(state_matrix_9), 'state_matrix_9')
        print(66, 135)
        key_matrix_9 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix_9, 'key_matrix_9')
        if type(key_matrix_9) == np.ndarray:
            print('malloc', sys.getsizeof(key_matrix_9), 'key_matrix_9',
                key_matrix_9.shape)
        elif type(key_matrix_9) == list:
            dims = []
            tmp = key_matrix_9
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(key_matrix_9), 'key_matrix_9', dims)
        elif type(key_matrix_9) == tuple:
            print('malloc', sys.getsizeof(key_matrix_9), 'key_matrix_9', [
                len(key_matrix_9)])
        else:
            print('malloc', sys.getsizeof(key_matrix_9), 'key_matrix_9')
        instrument_read(self, 'self').__sub_bytes(instrument_read(
            state_matrix_9, 'state_matrix_9'))
        instrument_read(self, 'self').__shift_rows(instrument_read(
            state_matrix_9, 'state_matrix_9'))
        instrument_read(self, 'self').__mix_columns(instrument_read(
            state_matrix_9, 'state_matrix_9'))
        instrument_read(self, 'self').__add_round_key(instrument_read(
            state_matrix_9, 'state_matrix_9'), instrument_read(key_matrix_9,
            'key_matrix_9'))
        print('exit scope 9')

    def __round_decrypt(self, state_matrix, key_matrix):
        print('enter scope 10')
        print(1, 141)
        print(69, 142)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(69, 143)
        state_matrix_10 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix_10, 'state_matrix_10')
        if type(state_matrix_10) == np.ndarray:
            print('malloc', sys.getsizeof(state_matrix_10),
                'state_matrix_10', state_matrix_10.shape)
        elif type(state_matrix_10) == list:
            dims = []
            tmp = state_matrix_10
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(state_matrix_10),
                'state_matrix_10', dims)
        elif type(state_matrix_10) == tuple:
            print('malloc', sys.getsizeof(state_matrix_10),
                'state_matrix_10', [len(state_matrix_10)])
        else:
            print('malloc', sys.getsizeof(state_matrix_10), 'state_matrix_10')
        print(69, 144)
        key_matrix_10 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix_10, 'key_matrix_10')
        if type(key_matrix_10) == np.ndarray:
            print('malloc', sys.getsizeof(key_matrix_10), 'key_matrix_10',
                key_matrix_10.shape)
        elif type(key_matrix_10) == list:
            dims = []
            tmp = key_matrix_10
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(key_matrix_10), 'key_matrix_10', dims
                )
        elif type(key_matrix_10) == tuple:
            print('malloc', sys.getsizeof(key_matrix_10), 'key_matrix_10',
                [len(key_matrix_10)])
        else:
            print('malloc', sys.getsizeof(key_matrix_10), 'key_matrix_10')
        instrument_read(self, 'self').__add_round_key(instrument_read(
            state_matrix_10, 'state_matrix_10'), instrument_read(
            key_matrix_10, 'key_matrix_10'))
        instrument_read(self, 'self').__inv_mix_columns(instrument_read(
            state_matrix_10, 'state_matrix_10'))
        instrument_read(self, 'self').__inv_shift_rows(instrument_read(
            state_matrix_10, 'state_matrix_10'))
        instrument_read(self, 'self').__inv_sub_bytes(instrument_read(
            state_matrix_10, 'state_matrix_10'))
        print('exit scope 10')

    def __sub_bytes(self, s):
        print('enter scope 11')
        print(1, 150)
        print(72, 151)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(72, 152)
        s_11 = instrument_read(s, 's')
        write_instrument_read(s_11, 's_11')
        if type(s_11) == np.ndarray:
            print('malloc', sys.getsizeof(s_11), 's_11', s_11.shape)
        elif type(s_11) == list:
            dims = []
            tmp = s_11
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_11), 's_11', dims)
        elif type(s_11) == tuple:
            print('malloc', sys.getsizeof(s_11), 's_11', [len(s_11)])
        else:
            print('malloc', sys.getsizeof(s_11), 's_11')
        for i_11 in range(4):
            instrument_read(loop, 'loop').start_unroll
            for j_11 in range(4):
                print(77, 156)
                s_11[instrument_read(instrument_read(i_11, 'i_11'), 'i_11')][
                    instrument_read(instrument_read(j_11, 'j_11'), 'j_11')
                    ] = instrument_read_sub(instrument_read(Sbox_0,
                    'Sbox_0'), 'Sbox_0', instrument_read_sub(
                    instrument_read_sub(instrument_read(s_11, 's_11'),
                    's_11', instrument_read(i_11, 'i_11'), None, None,
                    False), 's_11[i_11]', instrument_read(j_11, 'j_11'),
                    None, None, False), None, None, False)
                write_instrument_read_sub(s_11[instrument_read(
                    instrument_read(i_11, 'i_11'), 'i_11')],
                    "s_11[instrument_read(i_11, 'i_11')]", instrument_read(
                    instrument_read(j_11, 'j_11'), 'j_11'), None, None, False)
            instrument_read(loop, 'loop').stop_unroll
        print('exit scope 11')

    def __inv_sub_bytes(self, s):
        print('enter scope 12')
        print(1, 159)
        print(81, 160)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(81, 161)
        s_12 = instrument_read(s, 's')
        write_instrument_read(s_12, 's_12')
        if type(s_12) == np.ndarray:
            print('malloc', sys.getsizeof(s_12), 's_12', s_12.shape)
        elif type(s_12) == list:
            dims = []
            tmp = s_12
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_12), 's_12', dims)
        elif type(s_12) == tuple:
            print('malloc', sys.getsizeof(s_12), 's_12', [len(s_12)])
        else:
            print('malloc', sys.getsizeof(s_12), 's_12')
        for i_12 in range(4):
            instrument_read(loop, 'loop').start_unroll
            for j_12 in range(4):
                print(86, 165)
                s_12[instrument_read(instrument_read(i_12, 'i_12'), 'i_12')][
                    instrument_read(instrument_read(j_12, 'j_12'), 'j_12')
                    ] = instrument_read_sub(instrument_read(InvSbox_0,
                    'InvSbox_0'), 'InvSbox_0', instrument_read_sub(
                    instrument_read_sub(instrument_read(s_12, 's_12'),
                    's_12', instrument_read(i_12, 'i_12'), None, None,
                    False), 's_12[i_12]', instrument_read(j_12, 'j_12'),
                    None, None, False), None, None, False)
                write_instrument_read_sub(s_12[instrument_read(
                    instrument_read(i_12, 'i_12'), 'i_12')],
                    "s_12[instrument_read(i_12, 'i_12')]", instrument_read(
                    instrument_read(j_12, 'j_12'), 'j_12'), None, None, False)
            instrument_read(loop, 'loop').stop_unroll
        print('exit scope 12')

    def __shift_rows(self, s):
        print('enter scope 13')
        print(1, 168)
        print(90, 169)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(90, 170)
        s_13 = instrument_read(s, 's')
        write_instrument_read(s_13, 's_13')
        if type(s_13) == np.ndarray:
            print('malloc', sys.getsizeof(s_13), 's_13', s_13.shape)
        elif type(s_13) == list:
            dims = []
            tmp = s_13
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_13), 's_13', dims)
        elif type(s_13) == tuple:
            print('malloc', sys.getsizeof(s_13), 's_13', [len(s_13)])
        else:
            print('malloc', sys.getsizeof(s_13), 's_13')
        print(90, 171)
        print(90, 173)
        print(90, 175)
        print('exit scope 13')

    def __inv_shift_rows(self, s):
        print('enter scope 14')
        print(1, 178)
        print(93, 179)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(93, 180)
        s_14 = instrument_read(s, 's')
        write_instrument_read(s_14, 's_14')
        if type(s_14) == np.ndarray:
            print('malloc', sys.getsizeof(s_14), 's_14', s_14.shape)
        elif type(s_14) == list:
            dims = []
            tmp = s_14
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_14), 's_14', dims)
        elif type(s_14) == tuple:
            print('malloc', sys.getsizeof(s_14), 's_14', [len(s_14)])
        else:
            print('malloc', sys.getsizeof(s_14), 's_14')
        print(93, 181)
        print(93, 183)
        print(93, 185)
        print('exit scope 14')

    def __mix_single_column(self, a):
        print('enter scope 15')
        print(1, 188)
        print(96, 189)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(96, 190)
        a_15 = instrument_read(a, 'a')
        write_instrument_read(a_15, 'a_15')
        if type(a_15) == np.ndarray:
            print('malloc', sys.getsizeof(a_15), 'a_15', a_15.shape)
        elif type(a_15) == list:
            dims = []
            tmp = a_15
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(a_15), 'a_15', dims)
        elif type(a_15) == tuple:
            print('malloc', sys.getsizeof(a_15), 'a_15', [len(a_15)])
        else:
            print('malloc', sys.getsizeof(a_15), 'a_15')
        print(96, 191)
        t_15 = instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 1, None, None, False) ^ instrument_read_sub(
            instrument_read(a_15, 'a_15'), 'a_15', 2, None, None, False
            ) ^ instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 
            3, None, None, False)
        write_instrument_read(t_15, 't_15')
        if type(t_15) == np.ndarray:
            print('malloc', sys.getsizeof(t_15), 't_15', t_15.shape)
        elif type(t_15) == list:
            dims = []
            tmp = t_15
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(t_15), 't_15', dims)
        elif type(t_15) == tuple:
            print('malloc', sys.getsizeof(t_15), 't_15', [len(t_15)])
        else:
            print('malloc', sys.getsizeof(t_15), 't_15')
        print(96, 192)
        u_15 = instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False)
        write_instrument_read(u_15, 'u_15')
        if type(u_15) == np.ndarray:
            print('malloc', sys.getsizeof(u_15), 'u_15', u_15.shape)
        elif type(u_15) == list:
            dims = []
            tmp = u_15
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(u_15), 'u_15', dims)
        elif type(u_15) == tuple:
            print('malloc', sys.getsizeof(u_15), 'u_15', [len(u_15)])
        else:
            print('malloc', sys.getsizeof(u_15), 'u_15')
        print(96, 193)
        a_15[0] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 1, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 0, None, None, False)
        print(96, 194)
        a_15[1] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 1,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 2, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 1, None, None, False)
        print(96, 195)
        a_15[2] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 2,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 3, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 2, None, None, False)
        print(96, 196)
        a_15[3] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 3,
            None, None, False) ^ instrument_read(u_15, 'u_15'))
        write_instrument_read_sub(a_15, 'a_15', 3, None, None, False)
        print('exit scope 15')

    def __mix_columns(self, s):
        print('enter scope 16')
        print(1, 198)
        print(99, 199)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(99, 200)
        s_16 = instrument_read(s, 's')
        write_instrument_read(s_16, 's_16')
        if type(s_16) == np.ndarray:
            print('malloc', sys.getsizeof(s_16), 's_16', s_16.shape)
        elif type(s_16) == list:
            dims = []
            tmp = s_16
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_16), 's_16', dims)
        elif type(s_16) == tuple:
            print('malloc', sys.getsizeof(s_16), 's_16', [len(s_16)])
        else:
            print('malloc', sys.getsizeof(s_16), 's_16')
        for i_16 in range(4):
            instrument_read(self, 'self').__mix_single_column(
                instrument_read_sub(instrument_read(s_16, 's_16'), 's_16',
                instrument_read(i_16, 'i_16'), None, None, False))
        print('exit scope 16')

    def __inv_mix_columns(self, s):
        print('enter scope 17')
        print(1, 204)
        print(105, 205)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        if type(self) == np.ndarray:
            print('malloc', sys.getsizeof(self), 'self', self.shape)
        elif type(self) == list:
            dims = []
            tmp = self
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(self), 'self', dims)
        elif type(self) == tuple:
            print('malloc', sys.getsizeof(self), 'self', [len(self)])
        else:
            print('malloc', sys.getsizeof(self), 'self')
        print(105, 206)
        s_17 = instrument_read(s, 's')
        write_instrument_read(s_17, 's_17')
        if type(s_17) == np.ndarray:
            print('malloc', sys.getsizeof(s_17), 's_17', s_17.shape)
        elif type(s_17) == list:
            dims = []
            tmp = s_17
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(s_17), 's_17', dims)
        elif type(s_17) == tuple:
            print('malloc', sys.getsizeof(s_17), 's_17', [len(s_17)])
        else:
            print('malloc', sys.getsizeof(s_17), 's_17')
        for i_17 in range(4):
            print(107, 208)
            u_17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 0, None, None,
                False) ^ instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 2, None, None,
                False)))
            write_instrument_read(u_17, 'u_17')
            if type(u_17) == np.ndarray:
                print('malloc', sys.getsizeof(u_17), 'u_17', u_17.shape)
            elif type(u_17) == list:
                dims = []
                tmp = u_17
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(u_17), 'u_17', dims)
            elif type(u_17) == tuple:
                print('malloc', sys.getsizeof(u_17), 'u_17', [len(u_17)])
            else:
                print('malloc', sys.getsizeof(u_17), 'u_17')
            print(107, 209)
            v_17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 1, None, None,
                False) ^ instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 3, None, None,
                False)))
            write_instrument_read(v_17, 'v_17')
            if type(v_17) == np.ndarray:
                print('malloc', sys.getsizeof(v_17), 'v_17', v_17.shape)
            elif type(v_17) == list:
                dims = []
                tmp = v_17
                while type(tmp) == list:
                    dims.append(len(tmp))
                    if len(tmp) > 0:
                        tmp = tmp[0]
                    else:
                        tmp = None
                print('malloc', sys.getsizeof(v_17), 'v_17', dims)
            elif type(v_17) == tuple:
                print('malloc', sys.getsizeof(v_17), 'v_17', [len(v_17)])
            else:
                print('malloc', sys.getsizeof(v_17), 'v_17')
            print(107, 210)
            s_17[instrument_read(i_17, 'i_17')][0] ^= instrument_read(u_17,
                'u_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 0, None, None, False)
            print(107, 211)
            s_17[instrument_read(i_17, 'i_17')][1] ^= instrument_read(v_17,
                'v_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 1, None, None, False)
            print(107, 212)
            s_17[instrument_read(i_17, 'i_17')][2] ^= instrument_read(u_17,
                'u_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 2, None, None, False)
            print(107, 213)
            s_17[instrument_read(i_17, 'i_17')][3] ^= instrument_read(v_17,
                'v_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 3, None, None, False)
        instrument_read(self, 'self').__mix_columns(instrument_read(s_17,
            's_17'))
        print('exit scope 17')


if instrument_read(__name__, '__name__') == '__main__':
    import time
    print(110, 219)
    start_0 = instrument_read(time, 'time').time()
    write_instrument_read(start_0, 'start_0')
    if type(start_0) == np.ndarray:
        print('malloc', sys.getsizeof(start_0), 'start_0', start_0.shape)
    elif type(start_0) == list:
        dims = []
        tmp = start_0
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(start_0), 'start_0', dims)
    elif type(start_0) == tuple:
        print('malloc', sys.getsizeof(start_0), 'start_0', [len(start_0)])
    else:
        print('malloc', sys.getsizeof(start_0), 'start_0')
    for i_0 in range(10):
        print(113, 221)
        Sbox_0 += instrument_read(Sbox_0, 'Sbox_0')
        write_instrument_read(Sbox_0, 'Sbox_0')
    print(114, 222)
    Sbox_new_0 = instrument_read(Sbox_0, 'Sbox_0')
    write_instrument_read(Sbox_new_0, 'Sbox_new_0')
    if type(Sbox_new_0) == np.ndarray:
        print('malloc', sys.getsizeof(Sbox_new_0), 'Sbox_new_0', Sbox_new_0
            .shape)
    elif type(Sbox_new_0) == list:
        dims = []
        tmp = Sbox_new_0
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(Sbox_new_0), 'Sbox_new_0', dims)
    elif type(Sbox_new_0) == tuple:
        print('malloc', sys.getsizeof(Sbox_new_0), 'Sbox_new_0', [len(
            Sbox_new_0)])
    else:
        print('malloc', sys.getsizeof(Sbox_new_0), 'Sbox_new_0')
    for i_0 in range(1):
        print(116, 224)
        aes_0 = AES(1212304810341341)
        write_instrument_read(aes_0, 'aes_0')
        if type(aes_0) == np.ndarray:
            print('malloc', sys.getsizeof(aes_0), 'aes_0', aes_0.shape)
        elif type(aes_0) == list:
            dims = []
            tmp = aes_0
            while type(tmp) == list:
                dims.append(len(tmp))
                if len(tmp) > 0:
                    tmp = tmp[0]
                else:
                    tmp = None
            print('malloc', sys.getsizeof(aes_0), 'aes_0', dims)
        elif type(aes_0) == tuple:
            print('malloc', sys.getsizeof(aes_0), 'aes_0', [len(aes_0)])
        else:
            print('malloc', sys.getsizeof(aes_0), 'aes_0')
        instrument_read(aes_0, 'aes_0').encrypt(1212304810341341)
    print(117, 226)
    end_0 = instrument_read(time, 'time').time()
    write_instrument_read(end_0, 'end_0')
    if type(end_0) == np.ndarray:
        print('malloc', sys.getsizeof(end_0), 'end_0', end_0.shape)
    elif type(end_0) == list:
        dims = []
        tmp = end_0
        while type(tmp) == list:
            dims.append(len(tmp))
            if len(tmp) > 0:
                tmp = tmp[0]
            else:
                tmp = None
        print('malloc', sys.getsizeof(end_0), 'end_0', dims)
    elif type(end_0) == tuple:
        print('malloc', sys.getsizeof(end_0), 'end_0', [len(end_0)])
    else:
        print('malloc', sys.getsizeof(end_0), 'end_0')
