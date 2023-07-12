import sys
from instrument_lib import *
import sys
from instrument_lib import *
print(1, 1)
Sbox__0 = (99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215,
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
write_instrument_read(Sbox__0, 'Sbox__0')
print('malloc', sys.getsizeof(Sbox__0), 'Sbox__0')
print(1, 260)
InvSbox__0 = (82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243,
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
write_instrument_read(InvSbox__0, 'InvSbox__0')
print('malloc', sys.getsizeof(InvSbox__0), 'InvSbox__0')


def xtime(a):
    print('enter scope 1')
    print(1, 521)
    a__1 = instrument_read(a, 'a')
    write_instrument_read(a__1, 'a__1')
    print('malloc', sys.getsizeof(a__1), 'a__1')
    print('exit scope 1')
    return (instrument_read(a__1, 'a__1') << 1 ^ 27) & 255 if instrument_read(
        a__1, 'a__1') & 128 else instrument_read(a__1, 'a__1') << 1
    print('exit scope 1')


print(1, 525)
Rcon__0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 
    47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
write_instrument_read(Rcon__0, 'Rcon__0')
print('malloc', sys.getsizeof(Rcon__0), 'Rcon__0')


def text2matrix(text):
    print('enter scope 2')
    print(1, 561)
    text__2 = instrument_read(text, 'text')
    write_instrument_read(text__2, 'text__2')
    print('malloc', sys.getsizeof(text__2), 'text__2')
    print(7, 562)
    matrix__2 = []
    write_instrument_read(matrix__2, 'matrix__2')
    print('malloc', sys.getsizeof(matrix__2), 'matrix__2')
    for i__2 in range(16):
        print(9, 564)
        byte__2 = instrument_read(text__2, 'text__2') >> 8 * (15 -
            instrument_read(i__2, 'i__2')) & 255
        write_instrument_read(byte__2, 'byte__2')
        print('malloc', sys.getsizeof(byte__2), 'byte__2')
        if instrument_read(i__2, 'i__2') % 4 == 0:
            instrument_read(matrix__2, 'matrix__2').append([instrument_read
                (byte__2, 'byte__2')])
        else:
            instrument_read_sub(instrument_read(matrix__2, 'matrix__2'),
                'matrix__2', int(instrument_read(i__2, 'i__2') / 4), None,
                None, False).append(instrument_read(byte__2, 'byte__2'))
    print('exit scope 2')
    return instrument_read(matrix__2, 'matrix__2')
    print('exit scope 2')


def matrix2text(matrix):
    print('enter scope 3')
    print(1, 572)
    matrix__3 = instrument_read(matrix, 'matrix')
    write_instrument_read(matrix__3, 'matrix__3')
    print('malloc', sys.getsizeof(matrix__3), 'matrix__3')
    print(17, 573)
    text__3 = 0
    write_instrument_read(text__3, 'text__3')
    print('malloc', sys.getsizeof(text__3), 'text__3')
    for i__3 in range(4):
        for j__3 in range(4):
            print(21, 576)
            text__3 |= instrument_read_sub(instrument_read_sub(
                instrument_read(matrix__3, 'matrix__3'), 'matrix__3',
                instrument_read(i__3, 'i__3'), None, None, False),
                'matrix__3[i__3]', instrument_read(j__3, 'j__3'), None,
                None, False) << 120 - 8 * (4 * instrument_read(i__3, 'i__3'
                ) + instrument_read(j__3, 'j__3'))
            write_instrument_read(text__3, 'text__3')
    print('exit scope 3')
    return instrument_read(text__3, 'text__3')
    print('exit scope 3')


class AES:

    def __init__(self, master_key):
        print('enter scope 4')
        print(1, 581)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        master_key__4 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key__4, 'master_key__4')
        print('malloc', sys.getsizeof(master_key__4), 'master_key__4')
        instrument_read(self, 'self').change_key(instrument_read(
            master_key__4, 'master_key__4'))
        print('exit scope 4')

    def change_key(self, master_key):
        print('enter scope 5')
        print(1, 584)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        master_key__5 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key__5, 'master_key__5')
        print('malloc', sys.getsizeof(master_key__5), 'master_key__5')
        print(29, 585)
        instrument_read(self, 'self').round_keys = text2matrix(instrument_read
            (master_key__5, 'master_key__5'))
        for i__5 in range(4, 4 * 11):
            instrument_read(self, 'self').round_keys.append([])
            if instrument_read(i__5, 'i__5') % 4 == 0:
                print(33, 591)
                byte__5 = instrument_read_sub(instrument_read_sub(
                    instrument_read(self, 'self').round_keys,
                    'self.round_keys', instrument_read(i__5, 'i__5') - 4,
                    None, None, False), 'self.round_keys[i__5 - 4]', 0,
                    None, None, False) ^ instrument_read_sub(instrument_read
                    (Sbox__0, 'Sbox__0'), 'Sbox__0', instrument_read_sub(
                    instrument_read_sub(instrument_read(self, 'self').
                    round_keys, 'self.round_keys', instrument_read(i__5,
                    'i__5') - 1, None, None, False),
                    'self.round_keys[i__5 - 1]', 1, None, None, False),
                    None, None, False) ^ instrument_read_sub(instrument_read
                    (Rcon__0, 'Rcon__0'), 'Rcon__0', int(instrument_read(
                    i__5, 'i__5') / 4), None, None, False)
                write_instrument_read(byte__5, 'byte__5')
                print('malloc', sys.getsizeof(byte__5), 'byte__5')
                instrument_read_sub(instrument_read(self, 'self').
                    round_keys, 'self.round_keys', instrument_read(i__5,
                    'i__5'), None, None, False).append(instrument_read(
                    byte__5, 'byte__5'))
                for j__5 in range(1, 4):
                    print(39, 599)
                    byte__5 = instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i__5, 'i__5') - 
                        4, None, None, False), 'self.round_keys[i__5 - 4]',
                        instrument_read(j__5, 'j__5'), None, None, False
                        ) ^ instrument_read_sub(instrument_read(Sbox__0,
                        'Sbox__0'), 'Sbox__0', instrument_read_sub(
                        instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i__5,
                        'i__5') - 1, None, None, False),
                        'self.round_keys[i__5 - 1]', (instrument_read(j__5,
                        'j__5') + 1) % 4, None, None, False), None, None, False
                        )
                    write_instrument_read(byte__5, 'byte__5')
                    print('malloc', sys.getsizeof(byte__5), 'byte__5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i__5,
                        'i__5'), None, None, False).append(instrument_read(
                        byte__5, 'byte__5'))
            else:
                for j__5 in range(4):
                    print(36, 606)
                    byte__5 = instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i__5, 'i__5') - 
                        4, None, None, False), 'self.round_keys[i__5 - 4]',
                        instrument_read(j__5, 'j__5'), None, None, False
                        ) ^ instrument_read_sub(instrument_read_sub(
                        instrument_read(self, 'self').round_keys,
                        'self.round_keys', instrument_read(i__5, 'i__5') - 
                        1, None, None, False), 'self.round_keys[i__5 - 1]',
                        instrument_read(j__5, 'j__5'), None, None, False)
                    write_instrument_read(byte__5, 'byte__5')
                    print('malloc', sys.getsizeof(byte__5), 'byte__5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i__5,
                        'i__5'), None, None, False).append(instrument_read(
                        byte__5, 'byte__5'))
        print('exit scope 5')

    def encrypt(self, plaintext):
        print('enter scope 6')
        print(1, 611)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        plaintext__6 = instrument_read(plaintext, 'plaintext')
        write_instrument_read(plaintext__6, 'plaintext__6')
        print('malloc', sys.getsizeof(plaintext__6), 'plaintext__6')
        print(43, 612)
        instrument_read(self, 'self').plain_state = text2matrix(instrument_read
            (plaintext__6, 'plaintext__6'))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').plain_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, None, 4, True))
        for i__6 in range(1, 10):
            instrument_read(self, 'self').__round_encrypt(instrument_read(
                self, 'self').plain_state, instrument_read_sub(
                instrument_read(self, 'self').round_keys, 'self.round_keys',
                None, 4 * instrument_read(i__6, 'i__6'),
                4 * (instrument_read(i__6, 'i__6') + 1), True))
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
        print(1, 625)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        ciphertext__7 = instrument_read(ciphertext, 'ciphertext')
        write_instrument_read(ciphertext__7, 'ciphertext__7')
        print('malloc', sys.getsizeof(ciphertext__7), 'ciphertext__7')
        print(50, 626)
        instrument_read(self, 'self').cipher_state = text2matrix(
            instrument_read(ciphertext__7, 'ciphertext__7'))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').cipher_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, 40, None, True))
        instrument_read(self, 'self').__inv_shift_rows(instrument_read(self,
            'self').cipher_state)
        instrument_read(self, 'self').__inv_sub_bytes(instrument_read(self,
            'self').cipher_state)
        for i__7 in range(9, 0, -1):
            instrument_read(self, 'self').__round_decrypt(instrument_read(
                self, 'self').cipher_state, instrument_read_sub(
                instrument_read(self, 'self').round_keys, 'self.round_keys',
                None, 4 * instrument_read(i__7, 'i__7'),
                4 * (instrument_read(i__7, 'i__7') + 1), True))
        instrument_read(self, 'self').__add_round_key(instrument_read(self,
            'self').cipher_state, instrument_read_sub(instrument_read(self,
            'self').round_keys, 'self.round_keys', None, None, 4, True))
        print('exit scope 7')
        return matrix2text(instrument_read(self, 'self').cipher_state)
        print('exit scope 7')

    def __add_round_key(self, s, k):
        print('enter scope 8')
        print(1, 641)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__8 = instrument_read(s, 's')
        write_instrument_read(s__8, 's__8')
        print('malloc', sys.getsizeof(s__8), 's__8')
        k__8 = instrument_read(k, 'k')
        write_instrument_read(k__8, 'k__8')
        print('malloc', sys.getsizeof(k__8), 'k__8')
        for i__8 in range(4):
            for j__8 in range(4):
                print(60, 644)
                s__8[instrument_read(i__8, 'i__8')][instrument_read(j__8,
                    'j__8')] ^= instrument_read_sub(instrument_read_sub(
                    instrument_read(k__8, 'k__8'), 'k__8', instrument_read(
                    i__8, 'i__8'), None, None, False), 'k__8[i__8]',
                    instrument_read(j__8, 'j__8'), None, None, False)
                write_instrument_read_sub(s__8[instrument_read(
                    instrument_read(i__8, 'i__8'), 'i__8')],
                    "s__8[instrument_read(i__8, 'i__8')]", instrument_read(
                    instrument_read(j__8, 'j__8'), 'j__8'), None, None, False)
        print('exit scope 8')

    def __round_encrypt(self, state_matrix, key_matrix):
        print('enter scope 9')
        print(1, 646)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        state_matrix__9 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix__9, 'state_matrix__9')
        print('malloc', sys.getsizeof(state_matrix__9), 'state_matrix__9')
        key_matrix__9 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix__9, 'key_matrix__9')
        print('malloc', sys.getsizeof(key_matrix__9), 'key_matrix__9')
        instrument_read(self, 'self').__sub_bytes(instrument_read(
            state_matrix__9, 'state_matrix__9'))
        instrument_read(self, 'self').__shift_rows(instrument_read(
            state_matrix__9, 'state_matrix__9'))
        instrument_read(self, 'self').__mix_columns(instrument_read(
            state_matrix__9, 'state_matrix__9'))
        instrument_read(self, 'self').__add_round_key(instrument_read(
            state_matrix__9, 'state_matrix__9'), instrument_read(
            key_matrix__9, 'key_matrix__9'))
        print('exit scope 9')

    def __round_decrypt(self, state_matrix, key_matrix):
        print('enter scope 10')
        print(1, 652)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        state_matrix__10 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix__10, 'state_matrix__10')
        print('malloc', sys.getsizeof(state_matrix__10), 'state_matrix__10')
        key_matrix__10 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix__10, 'key_matrix__10')
        print('malloc', sys.getsizeof(key_matrix__10), 'key_matrix__10')
        instrument_read(self, 'self').__add_round_key(instrument_read(
            state_matrix__10, 'state_matrix__10'), instrument_read(
            key_matrix__10, 'key_matrix__10'))
        instrument_read(self, 'self').__inv_mix_columns(instrument_read(
            state_matrix__10, 'state_matrix__10'))
        instrument_read(self, 'self').__inv_shift_rows(instrument_read(
            state_matrix__10, 'state_matrix__10'))
        instrument_read(self, 'self').__inv_sub_bytes(instrument_read(
            state_matrix__10, 'state_matrix__10'))
        print('exit scope 10')

    def __sub_bytes(self, s):
        print('enter scope 11')
        print(1, 658)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__11 = instrument_read(s, 's')
        write_instrument_read(s__11, 's__11')
        print('malloc', sys.getsizeof(s__11), 's__11')
        for i__11 in range(4):
            for j__11 in range(4):
                print(73, 661)
                s__11[instrument_read(instrument_read(i__11, 'i__11'), 'i__11')
                    ][instrument_read(instrument_read(j__11, 'j__11'), 'j__11')
                    ] = instrument_read_sub(instrument_read(Sbox__0,
                    'Sbox__0'), 'Sbox__0', instrument_read_sub(
                    instrument_read_sub(instrument_read(s__11, 's__11'),
                    's__11', instrument_read(i__11, 'i__11'), None, None,
                    False), 's__11[i__11]', instrument_read(j__11, 'j__11'),
                    None, None, False), None, None, False)
                write_instrument_read_sub(s__11[instrument_read(
                    instrument_read(i__11, 'i__11'), 'i__11')],
                    "s__11[instrument_read(i__11, 'i__11')]",
                    instrument_read(instrument_read(j__11, 'j__11'),
                    'j__11'), None, None, False)
        print('exit scope 11')

    def __inv_sub_bytes(self, s):
        print('enter scope 12')
        print(1, 663)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__12 = instrument_read(s, 's')
        write_instrument_read(s__12, 's__12')
        print('malloc', sys.getsizeof(s__12), 's__12')
        for i__12 in range(4):
            for j__12 in range(4):
                print(80, 666)
                s__12[instrument_read(instrument_read(i__12, 'i__12'), 'i__12')
                    ][instrument_read(instrument_read(j__12, 'j__12'), 'j__12')
                    ] = instrument_read_sub(instrument_read(InvSbox__0,
                    'InvSbox__0'), 'InvSbox__0', instrument_read_sub(
                    instrument_read_sub(instrument_read(s__12, 's__12'),
                    's__12', instrument_read(i__12, 'i__12'), None, None,
                    False), 's__12[i__12]', instrument_read(j__12, 'j__12'),
                    None, None, False), None, None, False)
                write_instrument_read_sub(s__12[instrument_read(
                    instrument_read(i__12, 'i__12'), 'i__12')],
                    "s__12[instrument_read(i__12, 'i__12')]",
                    instrument_read(instrument_read(j__12, 'j__12'),
                    'j__12'), None, None, False)
        print('exit scope 12')

    def __shift_rows(self, s):
        print('enter scope 13')
        print(1, 668)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__13 = instrument_read(s, 's')
        write_instrument_read(s__13, 's__13')
        print('malloc', sys.getsizeof(s__13), 's__13')
        print(84, 669)
        print(84, 670)
        print(84, 671)
        print('exit scope 13')

    def __inv_shift_rows(self, s):
        print('enter scope 14')
        print(1, 673)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__14 = instrument_read(s, 's')
        write_instrument_read(s__14, 's__14')
        print('malloc', sys.getsizeof(s__14), 's__14')
        print(87, 674)
        print(87, 675)
        print(87, 676)
        print('exit scope 14')

    def __mix_single_column(self, a):
        print('enter scope 15')
        print(1, 678)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        a__15 = instrument_read(a, 'a')
        write_instrument_read(a__15, 'a__15')
        print('malloc', sys.getsizeof(a__15), 'a__15')
        print(90, 680)
        t__15 = instrument_read_sub(instrument_read(a__15, 'a__15'),
            'a__15', 0, None, None, False) ^ instrument_read_sub(
            instrument_read(a__15, 'a__15'), 'a__15', 1, None, None, False
            ) ^ instrument_read_sub(instrument_read(a__15, 'a__15'),
            'a__15', 2, None, None, False) ^ instrument_read_sub(
            instrument_read(a__15, 'a__15'), 'a__15', 3, None, None, False)
        write_instrument_read(t__15, 't__15')
        print('malloc', sys.getsizeof(t__15), 't__15')
        print(90, 681)
        u__15 = instrument_read_sub(instrument_read(a__15, 'a__15'),
            'a__15', 0, None, None, False)
        write_instrument_read(u__15, 'u__15')
        print('malloc', sys.getsizeof(u__15), 'u__15')
        print(90, 682)
        a__15[0] ^= instrument_read(t__15, 't__15') ^ xtime(
            instrument_read_sub(instrument_read(a__15, 'a__15'), 'a__15', 0,
            None, None, False) ^ instrument_read_sub(instrument_read(a__15,
            'a__15'), 'a__15', 1, None, None, False))
        write_instrument_read_sub(a__15, 'a__15', 0, None, None, False)
        print(90, 683)
        a__15[1] ^= instrument_read(t__15, 't__15') ^ xtime(
            instrument_read_sub(instrument_read(a__15, 'a__15'), 'a__15', 1,
            None, None, False) ^ instrument_read_sub(instrument_read(a__15,
            'a__15'), 'a__15', 2, None, None, False))
        write_instrument_read_sub(a__15, 'a__15', 1, None, None, False)
        print(90, 684)
        a__15[2] ^= instrument_read(t__15, 't__15') ^ xtime(
            instrument_read_sub(instrument_read(a__15, 'a__15'), 'a__15', 2,
            None, None, False) ^ instrument_read_sub(instrument_read(a__15,
            'a__15'), 'a__15', 3, None, None, False))
        write_instrument_read_sub(a__15, 'a__15', 2, None, None, False)
        print(90, 685)
        a__15[3] ^= instrument_read(t__15, 't__15') ^ xtime(
            instrument_read_sub(instrument_read(a__15, 'a__15'), 'a__15', 3,
            None, None, False) ^ instrument_read(u__15, 'u__15'))
        write_instrument_read_sub(a__15, 'a__15', 3, None, None, False)
        print('exit scope 15')

    def __mix_columns(self, s):
        print('enter scope 16')
        print(1, 687)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__16 = instrument_read(s, 's')
        write_instrument_read(s__16, 's__16')
        print('malloc', sys.getsizeof(s__16), 's__16')
        for i__16 in range(4):
            instrument_read(self, 'self').__mix_single_column(
                instrument_read_sub(instrument_read(s__16, 's__16'),
                's__16', instrument_read(i__16, 'i__16'), None, None, False))
        print('exit scope 16')

    def __inv_mix_columns(self, s):
        print('enter scope 17')
        print(1, 691)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s__17 = instrument_read(s, 's')
        write_instrument_read(s__17, 's__17')
        print('malloc', sys.getsizeof(s__17), 's__17')
        for i__17 in range(4):
            print(99, 694)
            u__17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s__17, 's__17'), 's__17', instrument_read(
                i__17, 'i__17'), None, None, False), 's__17[i__17]', 0,
                None, None, False) ^ instrument_read_sub(
                instrument_read_sub(instrument_read(s__17, 's__17'),
                's__17', instrument_read(i__17, 'i__17'), None, None, False
                ), 's__17[i__17]', 2, None, None, False)))
            write_instrument_read(u__17, 'u__17')
            print('malloc', sys.getsizeof(u__17), 'u__17')
            print(99, 695)
            v__17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s__17, 's__17'), 's__17', instrument_read(
                i__17, 'i__17'), None, None, False), 's__17[i__17]', 1,
                None, None, False) ^ instrument_read_sub(
                instrument_read_sub(instrument_read(s__17, 's__17'),
                's__17', instrument_read(i__17, 'i__17'), None, None, False
                ), 's__17[i__17]', 3, None, None, False)))
            write_instrument_read(v__17, 'v__17')
            print('malloc', sys.getsizeof(v__17), 'v__17')
            print(99, 696)
            s__17[instrument_read(i__17, 'i__17')][0] ^= instrument_read(u__17,
                'u__17')
            write_instrument_read_sub(s__17[instrument_read(instrument_read
                (i__17, 'i__17'), 'i__17')],
                "s__17[instrument_read(i__17, 'i__17')]", 0, None, None, False)
            print(99, 697)
            s__17[instrument_read(i__17, 'i__17')][1] ^= instrument_read(v__17,
                'v__17')
            write_instrument_read_sub(s__17[instrument_read(instrument_read
                (i__17, 'i__17'), 'i__17')],
                "s__17[instrument_read(i__17, 'i__17')]", 1, None, None, False)
            print(99, 698)
            s__17[instrument_read(i__17, 'i__17')][2] ^= instrument_read(u__17,
                'u__17')
            write_instrument_read_sub(s__17[instrument_read(instrument_read
                (i__17, 'i__17'), 'i__17')],
                "s__17[instrument_read(i__17, 'i__17')]", 2, None, None, False)
            print(99, 699)
            s__17[instrument_read(i__17, 'i__17')][3] ^= instrument_read(v__17,
                'v__17')
            write_instrument_read_sub(s__17[instrument_read(instrument_read
                (i__17, 'i__17'), 'i__17')],
                "s__17[instrument_read(i__17, 'i__17')]", 3, None, None, False)
        instrument_read(self, 'self').__mix_columns(instrument_read(s__17,
            's__17'))
        print('exit scope 17')


if instrument_read(__name__, '__name__') == '__main__':
    import time
    print(102, 705)
    start__0 = instrument_read(time, 'time').time()
    write_instrument_read(start__0, 'start__0')
    print('malloc', sys.getsizeof(start__0), 'start__0')
    print(102, 706)
    aes__0 = AES(1212304810341341)
    write_instrument_read(aes__0, 'aes__0')
    print('malloc', sys.getsizeof(aes__0), 'aes__0')
    instrument_read(aes__0, 'aes__0').encrypt(1212304810341341)
    print(102, 708)
    end__0 = instrument_read(time, 'time').time()
    write_instrument_read(end__0, 'end__0')
    print('malloc', sys.getsizeof(end__0), 'end__0')
