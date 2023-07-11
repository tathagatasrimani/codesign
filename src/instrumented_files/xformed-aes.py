import sys
from instrument_lib import *
print(1, 1)
Sbox0 = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 
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
write_instrument_read(Sbox0, 'Sbox0')
print('malloc', sys.getsizeof(Sbox0), 'Sbox0')
print(1, 260)
InvSbox0 = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 
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
write_instrument_read(InvSbox0, 'InvSbox0')
print('malloc', sys.getsizeof(InvSbox0), 'InvSbox0')


def xtime(a1):
    print('enter scope 1')
    print(1, 521)
    return (instrument_read(a1, 'a1') << 1 ^ 27) & 255 if instrument_read(a1,
        'a1') & 128 else instrument_read(a1, 'a1') << 1
    print('exit scope 1')


print(1, 525)
Rcon0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47,
    94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
write_instrument_read(Rcon0, 'Rcon0')
print('malloc', sys.getsizeof(Rcon0), 'Rcon0')


def text2matrix(text2):
    print('enter scope 2')
    print(1, 561)
    print(7, 562)
    matrix2 = []
    write_instrument_read(matrix2, 'matrix2')
    print('malloc', sys.getsizeof(matrix2), 'matrix2')
    print('enter scope 3')
    for i3 in range(16):
        byte3 = instrument_read(text3, 'text3') >> 8 * (15 -
            instrument_read(i3, 'i3')) & 255
        if instrument_read(i3, 'i3') % 4 == 0:
            matrix.append([byte])
        else:
            matrix[int(i / 4)].append(byte)
    print('exit scope 3')
    return instrument_read(matrix2, 'matrix2')
    print('exit scope 2')


def matrix2text(matrix5):
    print('enter scope 5')
    print(1, 572)
    print(17, 573)
    text5 = 0
    write_instrument_read(text5, 'text5')
    print('malloc', sys.getsizeof(text5), 'text5')
    print('enter scope 6')
    for i6 in range(4):
        for j in range(4):
            text5 |= instrument_read_sub(instrument_read_sub(
                instrument_read(matrix7, 'matrix7'), 'matrix',
                instrument_read(i6, 'i6')), 'matrix[i]', instrument_read(j7,
                'j7')) << 120 - 8 * (4 * instrument_read(i6, 'i6') +
                instrument_read(j7, 'j7'))
    print('exit scope 6')
    return instrument_read(text5, 'text5')
    print('exit scope 5')


class AES:

    def __init__(self, master_key8):
        print('enter scope 8')
        print(1, 581)
        self.change_key(master_key)
        print('exit scope 8')

    def change_key(self, master_key9):
        print('enter scope 9')
        print(1, 584)
        print(29, 585)
        self.round_keys = text2matrix(master_key)
        print('enter scope 10')
        for i10 in range(4, 4 * 11):
            self.round_keys.append([])
            if instrument_read(i10, 'i10') % 4 == 0:
                byte11 = instrument_read_sub(instrument_read_sub(self.
                    round_keys, 'self.round_keys', instrument_read(i10,
                    'i10') - 4),
                    "self.round_keys[instrument_read(i10, 'i10') - 4]", 0
                    ) ^ instrument_read_sub(instrument_read(Sbox0, 'Sbox0'),
                    'Sbox', instrument_read_sub(instrument_read_sub(self.
                    round_keys, 'self.round_keys', instrument_read(i10,
                    'i10') - 1),
                    "self.round_keys[instrument_read(i10, 'i10') - 1]", 1)
                    ) ^ instrument_read_sub(instrument_read(Rcon0, 'Rcon0'),
                    'Rcon', int(i / 4))
                self.round_keys[i].append(byte)
                for j in range(1, 4):
                    byte11 = instrument_read_sub(instrument_read_sub(self.
                        round_keys, 'self.round_keys', instrument_read(i10,
                        'i10') - 4),
                        "self.round_keys[instrument_read(i10, 'i10') - 4]",
                        instrument_read(j12, 'j12')) ^ instrument_read_sub(
                        instrument_read(Sbox0, 'Sbox0'), 'Sbox',
                        instrument_read_sub(instrument_read_sub(self.
                        round_keys, 'self.round_keys', instrument_read(i10,
                        'i10') - 1),
                        "self.round_keys[instrument_read(i10, 'i10') - 1]",
                        (instrument_read(j12, 'j12') + 1) % 4))
                    self.round_keys[i].append(byte)
            else:
                for j in range(4):
                    byte11 = instrument_read_sub(instrument_read_sub(self.
                        round_keys, 'self.round_keys', instrument_read(i10,
                        'i10') - 4),
                        "self.round_keys[instrument_read(i10, 'i10') - 4]",
                        instrument_read(j13, 'j13')) ^ instrument_read_sub(
                        instrument_read_sub(self.round_keys,
                        'self.round_keys', instrument_read(i10, 'i10') - 1),
                        "self.round_keys[instrument_read(i10, 'i10') - 1]",
                        instrument_read(j13, 'j13'))
                    self.round_keys[i].append(byte)
        print('exit scope 10')
        print('exit scope 9')

    def encrypt(self, plaintext14):
        print('enter scope 14')
        print(1, 611)
        print(43, 612)
        self.plain_state = text2matrix(plaintext)
        self.__add_round_key(self.plain_state, self.round_keys[:4])
        print('enter scope 15')
        for i15 in range(1, 10):
            self.__round_encrypt(self.plain_state, self.round_keys[4 * i:4 *
                (i + 1)])
        print('exit scope 15')
        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])
        return matrix2text(self.plain_state)
        print('exit scope 14')

    def decrypt(self, ciphertext16):
        print('enter scope 16')
        print(1, 625)
        print(50, 626)
        self.cipher_state = text2matrix(ciphertext)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)
        print('enter scope 17')
        for i17 in range(9, 0, -1):
            self.__round_decrypt(self.cipher_state, self.round_keys[4 * i:4 *
                (i + 1)])
        print('exit scope 17')
        self.__add_round_key(self.cipher_state, self.round_keys[:4])
        return matrix2text(self.cipher_state)
        print('exit scope 16')

    def __add_round_key(self, s18, k18):
        print('enter scope 18')
        print(1, 641)
        print('enter scope 19')
        for i19 in range(4):
            for j in range(4):
                s20[instrument_read(i19, 'i19')][instrument_read(j20, 'j20')
                    ] ^= instrument_read_sub(instrument_read_sub(
                    instrument_read(k20, 'k20'), 'k', instrument_read(i19,
                    'i19')), 'k[i]', instrument_read(j20, 'j20'))
        print('exit scope 19')
        print('exit scope 18')

    def __round_encrypt(self, state_matrix21, key_matrix21):
        print('enter scope 21')
        print(1, 646)
        self.__sub_bytes(state_matrix)
        self.__shift_rows(state_matrix)
        self.__mix_columns(state_matrix)
        self.__add_round_key(state_matrix, key_matrix)
        print('exit scope 21')

    def __round_decrypt(self, state_matrix22, key_matrix22):
        print('enter scope 22')
        print(1, 652)
        self.__add_round_key(state_matrix, key_matrix)
        self.__inv_mix_columns(state_matrix)
        self.__inv_shift_rows(state_matrix)
        self.__inv_sub_bytes(state_matrix)
        print('exit scope 22')

    def __sub_bytes(self, s23):
        print('enter scope 23')
        print(1, 658)
        print('enter scope 24')
        for i24 in range(4):
            for j in range(4):
                s25[instrument_read(i24, 'i24')][instrument_read(j25, 'j25')
                    ] = instrument_read_sub(instrument_read(Sbox0, 'Sbox0'),
                    'Sbox', instrument_read_sub(instrument_read_sub(
                    instrument_read(s25, 's25'), 's', instrument_read(i24,
                    'i24')), 's[i]', instrument_read(j25, 'j25')))
        print('exit scope 24')
        print('exit scope 23')

    def __inv_sub_bytes(self, s26):
        print('enter scope 26')
        print(1, 663)
        print('enter scope 27')
        for i27 in range(4):
            for j in range(4):
                s28[instrument_read(i27, 'i27')][instrument_read(j28, 'j28')
                    ] = instrument_read_sub(instrument_read(InvSbox0,
                    'InvSbox0'), 'InvSbox', instrument_read_sub(
                    instrument_read_sub(instrument_read(s28, 's28'), 's',
                    instrument_read(i27, 'i27')), 's[i]', instrument_read(
                    j28, 'j28')))
        print('exit scope 27')
        print('exit scope 26')

    def __shift_rows(self, s29):
        print('enter scope 29')
        print(1, 668)
        print('exit scope 29')

    def __inv_shift_rows(self, s30):
        print('enter scope 30')
        print(1, 673)
        print('exit scope 30')

    def __mix_single_column(self, a31):
        print('enter scope 31')
        print(1, 678)
        print(90, 680)
        t31 = instrument_read_sub(instrument_read(a31, 'a31'), 'a', 0
            ) ^ instrument_read_sub(instrument_read(a31, 'a31'), 'a', 1
            ) ^ instrument_read_sub(instrument_read(a31, 'a31'), 'a', 2
            ) ^ instrument_read_sub(instrument_read(a31, 'a31'), 'a', 3)
        write_instrument_read(t31, 't31')
        print('malloc', sys.getsizeof(t31), 't31')
        print(90, 681)
        u31 = instrument_read_sub(instrument_read(a31, 'a31'), 'a', 0)
        write_instrument_read(u31, 'u31')
        print('malloc', sys.getsizeof(u31), 'u31')
        print(90, 682)
        a[0] ^= instrument_read(t31, 't31') ^ xtime(a[0] ^ a[1])
        write_instrument_read_sub(a31, 'a', 0)
        print(90, 683)
        a[1] ^= instrument_read(t31, 't31') ^ xtime(a[1] ^ a[2])
        write_instrument_read_sub(a31, 'a', 1)
        print(90, 684)
        a[2] ^= instrument_read(t31, 't31') ^ xtime(a[2] ^ a[3])
        write_instrument_read_sub(a31, 'a', 2)
        print(90, 685)
        a[3] ^= instrument_read(t31, 't31') ^ xtime(a[3] ^ u)
        write_instrument_read_sub(a31, 'a', 3)
        print('exit scope 31')

    def __mix_columns(self, s32):
        print('enter scope 32')
        print(1, 687)
        print('enter scope 33')
        for i33 in range(4):
            self.__mix_single_column(s[i])
        print('exit scope 33')
        print('exit scope 32')

    def __inv_mix_columns(self, s34):
        print('enter scope 34')
        print(1, 691)
        print('enter scope 35')
        for i35 in range(4):
            u35 = xtime(xtime(s[i][0] ^ s[i][2]))
            v35 = xtime(xtime(s[i][1] ^ s[i][3]))
            s35[instrument_read(i35, 'i35')][0] ^= instrument_read(u35, 'u35')
            s35[instrument_read(i35, 'i35')][1] ^= instrument_read(v35, 'v35')
            s35[instrument_read(i35, 'i35')][2] ^= instrument_read(u35, 'u35')
            s35[instrument_read(i35, 'i35')][3] ^= instrument_read(v35, 'v35')
        print('exit scope 35')
        self.__mix_columns(s)
        print('exit scope 34')


print('enter scope 36')
if __name__ == '__main__':
    print(1, 703)
    import time
    print(102, 705)
    start36 = time.time()
    write_instrument_read(start36, 'start36')
    print('malloc', sys.getsizeof(start36), 'start36')
    print(102, 706)
    aes36 = AES(1212304810341341)
    write_instrument_read(aes36, 'aes36')
    print('malloc', sys.getsizeof(aes36), 'aes36')
    aes.encrypt(1212304810341341)
    print(102, 708)
    end36 = time.time()
    write_instrument_read(end36, 'end36')
    print('malloc', sys.getsizeof(end36), 'end36')
else:
    print(1, 703)
print('exit scope 36')
