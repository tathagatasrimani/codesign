import sys
from instrument_lib import *
import sys
from instrument_lib import *
from loop import loop
print(1, 2)
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
print('malloc', sys.getsizeof(Sbox_0), 'Sbox_0')
print(1, 261)
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
print('malloc', sys.getsizeof(InvSbox_0), 'InvSbox_0')


def xtime(a):
    print('enter scope 1')
    print(1, 522)
    a_1 = instrument_read(a, 'a')
    write_instrument_read(a_1, 'a_1')
    print('malloc', sys.getsizeof(a_1), 'a_1')
    print('exit scope 1')
    return (instrument_read(a_1, 'a_1') << 1 ^ 27) & 255 if instrument_read(a_1
        , 'a_1') & 128 else instrument_read(a_1, 'a_1') << 1
    print('exit scope 1')


print(1, 526)
Rcon_0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 
    47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)
write_instrument_read(Rcon_0, 'Rcon_0')
print('malloc', sys.getsizeof(Rcon_0), 'Rcon_0')


def text2matrix(text):
    print('enter scope 2')
    print(1, 562)
    text_2 = instrument_read(text, 'text')
    write_instrument_read(text_2, 'text_2')
    print('malloc', sys.getsizeof(text_2), 'text_2')
    print(7, 563)
    matrix_2 = []
    write_instrument_read(matrix_2, 'matrix_2')
    print('malloc', sys.getsizeof(matrix_2), 'matrix_2')
    for i_2 in range(16):
        print(9, 565)
        byte_2 = instrument_read(text_2, 'text_2') >> 8 * (15 -
            instrument_read(i_2, 'i_2')) & 255
        write_instrument_read(byte_2, 'byte_2')
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
    print(1, 573)
    matrix_3 = instrument_read(matrix, 'matrix')
    write_instrument_read(matrix_3, 'matrix_3')
    print('malloc', sys.getsizeof(matrix_3), 'matrix_3')
    print(17, 574)
    text_3 = 0
    write_instrument_read(text_3, 'text_3')
    print('malloc', sys.getsizeof(text_3), 'text_3')
    for i_3 in range(4):
        for j_3 in range(4):
            print(21, 577)
            text_3 |= instrument_read_sub(instrument_read_sub(
                instrument_read(matrix_3, 'matrix_3'), 'matrix_3',
                instrument_read(i_3, 'i_3'), None, None, False),
                'matrix_3[i_3]', instrument_read(j_3, 'j_3'), None, None, False
                ) << 120 - 8 * (4 * instrument_read(i_3, 'i_3') +
                instrument_read(j_3, 'j_3'))
            write_instrument_read(text_3, 'text_3')
    print('exit scope 3')
    return instrument_read(text_3, 'text_3')
    print('exit scope 3')


class AES:

    def __init__(self, master_key):
        print('enter scope 4')
        print(1, 582)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        master_key_4 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key_4, 'master_key_4')
        print('malloc', sys.getsizeof(master_key_4), 'master_key_4')
        instrument_read(self, 'self').change_key(instrument_read(
            master_key_4, 'master_key_4'))
        print('exit scope 4')

    def change_key(self, master_key):
        print('enter scope 5')
        print(1, 585)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        master_key_5 = instrument_read(master_key, 'master_key')
        write_instrument_read(master_key_5, 'master_key_5')
        print('malloc', sys.getsizeof(master_key_5), 'master_key_5')
        print(29, 586)
        instrument_read(self, 'self').round_keys = text2matrix(instrument_read
            (master_key_5, 'master_key_5'))
        for i_5 in range(4, 4 * 11):
            instrument_read(self, 'self').round_keys.append([])
            if instrument_read(i_5, 'i_5') % 4 == 0:
                print(33, 592)
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
                print('malloc', sys.getsizeof(byte_5), 'byte_5')
                instrument_read_sub(instrument_read(self, 'self').
                    round_keys, 'self.round_keys', instrument_read(i_5,
                    'i_5'), None, None, False).append(instrument_read(
                    byte_5, 'byte_5'))
                for j_5 in range(1, 4):
                    print(39, 600)
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
                    print('malloc', sys.getsizeof(byte_5), 'byte_5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i_5,
                        'i_5'), None, None, False).append(instrument_read(
                        byte_5, 'byte_5'))
            else:
                for j_5 in range(4):
                    print(36, 607)
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
                    print('malloc', sys.getsizeof(byte_5), 'byte_5')
                    instrument_read_sub(instrument_read(self, 'self').
                        round_keys, 'self.round_keys', instrument_read(i_5,
                        'i_5'), None, None, False).append(instrument_read(
                        byte_5, 'byte_5'))
        print('exit scope 5')

    def encrypt(self, plaintext):
        print('enter scope 6')
        print(1, 612)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        plaintext_6 = instrument_read(plaintext, 'plaintext')
        write_instrument_read(plaintext_6, 'plaintext_6')
        print('malloc', sys.getsizeof(plaintext_6), 'plaintext_6')
        print(43, 613)
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
        print(1, 626)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        ciphertext_7 = instrument_read(ciphertext, 'ciphertext')
        write_instrument_read(ciphertext_7, 'ciphertext_7')
        print('malloc', sys.getsizeof(ciphertext_7), 'ciphertext_7')
        print(50, 627)
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
        print(1, 642)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_8 = instrument_read(s, 's')
        write_instrument_read(s_8, 's_8')
        print('malloc', sys.getsizeof(s_8), 's_8')
        k_8 = instrument_read(k, 'k')
        write_instrument_read(k_8, 'k_8')
        print('malloc', sys.getsizeof(k_8), 'k_8')
        for i_8 in range(4):
            for j_8 in range(4):
                print(60, 645)
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
        print(1, 647)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        state_matrix_9 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix_9, 'state_matrix_9')
        print('malloc', sys.getsizeof(state_matrix_9), 'state_matrix_9')
        key_matrix_9 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix_9, 'key_matrix_9')
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
        print(1, 653)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        state_matrix_10 = instrument_read(state_matrix, 'state_matrix')
        write_instrument_read(state_matrix_10, 'state_matrix_10')
        print('malloc', sys.getsizeof(state_matrix_10), 'state_matrix_10')
        key_matrix_10 = instrument_read(key_matrix, 'key_matrix')
        write_instrument_read(key_matrix_10, 'key_matrix_10')
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
        print(1, 659)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_11 = instrument_read(s, 's')
        write_instrument_read(s_11, 's_11')
        print('malloc', sys.getsizeof(s_11), 's_11')
        for i_11 in range(4):
            for j_11 in range(4):
                print(73, 662)
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
        print('exit scope 11')

    def __inv_sub_bytes(self, s):
        print('enter scope 12')
        print(1, 664)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_12 = instrument_read(s, 's')
        write_instrument_read(s_12, 's_12')
        print('malloc', sys.getsizeof(s_12), 's_12')
        for i_12 in range(4):
            for j_12 in range(4):
                print(80, 667)
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
        print('exit scope 12')

    def __shift_rows(self, s):
        print('enter scope 13')
        print(1, 669)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_13 = instrument_read(s, 's')
        write_instrument_read(s_13, 's_13')
        print('malloc', sys.getsizeof(s_13), 's_13')
        print(84, 670)
        print(84, 671)
        print(84, 672)
        print('exit scope 13')

    def __inv_shift_rows(self, s):
        print('enter scope 14')
        print(1, 674)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_14 = instrument_read(s, 's')
        write_instrument_read(s_14, 's_14')
        print('malloc', sys.getsizeof(s_14), 's_14')
        print(87, 675)
        print(87, 676)
        print(87, 677)
        print('exit scope 14')

    def __mix_single_column(self, a):
        print('enter scope 15')
        print(1, 679)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        a_15 = instrument_read(a, 'a')
        write_instrument_read(a_15, 'a_15')
        print('malloc', sys.getsizeof(a_15), 'a_15')
        print(90, 681)
        t_15 = instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 1, None, None, False) ^ instrument_read_sub(
            instrument_read(a_15, 'a_15'), 'a_15', 2, None, None, False
            ) ^ instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 
            3, None, None, False)
        write_instrument_read(t_15, 't_15')
        print('malloc', sys.getsizeof(t_15), 't_15')
        print(90, 682)
        u_15 = instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False)
        write_instrument_read(u_15, 'u_15')
        print('malloc', sys.getsizeof(u_15), 'u_15')
        print(90, 683)
        a_15[0] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 0,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 1, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 0, None, None, False)
        print(90, 684)
        a_15[1] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 1,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 2, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 1, None, None, False)
        print(90, 685)
        a_15[2] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 2,
            None, None, False) ^ instrument_read_sub(instrument_read(a_15,
            'a_15'), 'a_15', 3, None, None, False))
        write_instrument_read_sub(a_15, 'a_15', 2, None, None, False)
        print(90, 686)
        a_15[3] ^= instrument_read(t_15, 't_15') ^ xtime(
            instrument_read_sub(instrument_read(a_15, 'a_15'), 'a_15', 3,
            None, None, False) ^ instrument_read(u_15, 'u_15'))
        write_instrument_read_sub(a_15, 'a_15', 3, None, None, False)
        print('exit scope 15')

    def __mix_columns(self, s):
        print('enter scope 16')
        print(1, 688)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_16 = instrument_read(s, 's')
        write_instrument_read(s_16, 's_16')
        print('malloc', sys.getsizeof(s_16), 's_16')
        for i_16 in range(4):
            instrument_read(self, 'self').__mix_single_column(
                instrument_read_sub(instrument_read(s_16, 's_16'), 's_16',
                instrument_read(i_16, 'i_16'), None, None, False))
        print('exit scope 16')

    def __inv_mix_columns(self, s):
        print('enter scope 17')
        print(1, 692)
        self = instrument_read(self, 'self')
        write_instrument_read(self, 'self')
        print('malloc', sys.getsizeof(self), 'self')
        s_17 = instrument_read(s, 's')
        write_instrument_read(s_17, 's_17')
        print('malloc', sys.getsizeof(s_17), 's_17')
        for i_17 in range(4):
            print(99, 695)
            u_17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 0, None, None,
                False) ^ instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 2, None, None,
                False)))
            write_instrument_read(u_17, 'u_17')
            print('malloc', sys.getsizeof(u_17), 'u_17')
            print(99, 696)
            v_17 = xtime(xtime(instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 1, None, None,
                False) ^ instrument_read_sub(instrument_read_sub(
                instrument_read(s_17, 's_17'), 's_17', instrument_read(i_17,
                'i_17'), None, None, False), 's_17[i_17]', 3, None, None,
                False)))
            write_instrument_read(v_17, 'v_17')
            print('malloc', sys.getsizeof(v_17), 'v_17')
            print(99, 697)
            s_17[instrument_read(i_17, 'i_17')][0] ^= instrument_read(u_17,
                'u_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 0, None, None, False)
            print(99, 698)
            s_17[instrument_read(i_17, 'i_17')][1] ^= instrument_read(v_17,
                'v_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 1, None, None, False)
            print(99, 699)
            s_17[instrument_read(i_17, 'i_17')][2] ^= instrument_read(u_17,
                'u_17')
            write_instrument_read_sub(s_17[instrument_read(instrument_read(
                i_17, 'i_17'), 'i_17')],
                "s_17[instrument_read(i_17, 'i_17')]", 2, None, None, False)
            print(99, 700)
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
    for i_0 in range(1):
        print(105, 707)
        start_0 = instrument_read(time, 'time').time()
        write_instrument_read(start_0, 'start_0')
        print('malloc', sys.getsizeof(start_0), 'start_0')
        for i_0 in range(10):
            print(108, 709)
            Sbox_0 += instrument_read(Sbox_0, 'Sbox_0')
            write_instrument_read(Sbox_0, 'Sbox_0')
        print(109, 710)
        Sbox_new_0 = instrument_read(Sbox_0, 'Sbox_0')
        write_instrument_read(Sbox_new_0, 'Sbox_new_0')
        print('malloc', sys.getsizeof(Sbox_new_0), 'Sbox_new_0')
        print(109, 711)
        aes_0 = AES(1212304810341341)
        write_instrument_read(aes_0, 'aes_0')
        print('malloc', sys.getsizeof(aes_0), 'aes_0')
        instrument_read(aes_0, 'aes_0').encrypt(1212304810341341)
        print(109, 713)
        end_0 = instrument_read(time, 'time').time()
        write_instrument_read(end_0, 'end_0')
        print('malloc', sys.getsizeof(end_0), 'end_0')
