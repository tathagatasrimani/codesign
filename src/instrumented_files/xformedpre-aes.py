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


def xtime(a):
    print('enter scope 1')
    print(1, 521)
    a__1 = a
    print('exit scope 1')
    return (a__1 << 1 ^ 27) & 255 if a__1 & 128 else a__1 << 1
    print('exit scope 1')


print(1, 525)
Rcon__0 = (0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 
    47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 57)


def text2matrix(text):
    print('enter scope 2')
    print(1, 561)
    text__2 = text
    print(7, 562)
    matrix__2 = []
    for i__2 in range(16):
        print('enter scope 3')
        print(9, 564)
        byte__3 = text__2 >> 8 * (15 - i__2) & 255
        print('enter scope 4')
        if i__2 % 4 == 0:
            matrix__2.append([byte__3])
        else:
            matrix__2[int(i__2 / 4)].append(byte__3)
        print('exit scope 4')
        print('exit scope 3')
    print('exit scope 2')
    return matrix__2
    print('exit scope 2')


def matrix2text(matrix):
    print('enter scope 5')
    print(1, 572)
    matrix__5 = matrix
    print(17, 573)
    text__5 = 0
    for i__5 in range(4):
        print('enter scope 6')
        for j__6 in range(4):
            print('enter scope 7')
            print(21, 576)
            text__5 |= matrix__5[i__5][j__6] << 120 - 8 * (4 * i__5 + j__6)
            print('exit scope 7')
        print('exit scope 6')
    print('exit scope 5')
    return text__5
    print('exit scope 5')


class AES:

    def __init__(self, master_key):
        print('enter scope 8')
        print(1, 581)
        self = self
        master_key__8 = master_key
        self.change_key(master_key__8)
        print('exit scope 8')

    def change_key(self, master_key):
        print('enter scope 9')
        print(1, 584)
        self = self
        master_key__9 = master_key
        print(29, 585)
        self.round_keys = text2matrix(master_key__9)
        for i__9 in range(4, 4 * 11):
            print('enter scope 10')
            self.round_keys.append([])
            print('enter scope 11')
            if i__9 % 4 == 0:
                print(33, 591)
                byte__11 = self.round_keys[i__9 - 4][0] ^ Sbox__0[self.
                    round_keys[i__9 - 1][1]] ^ Rcon__0[int(i__9 / 4)]
                self.round_keys[i__9].append(byte__11)
                for j__11 in range(1, 4):
                    print('enter scope 12')
                    print(39, 599)
                    byte__11 = self.round_keys[i__9 - 4][j__11] ^ Sbox__0[self
                        .round_keys[i__9 - 1][(j__11 + 1) % 4]]
                    self.round_keys[i__9].append(byte__11)
                    print('exit scope 12')
            else:
                for j__11 in range(4):
                    print('enter scope 13')
                    print(36, 606)
                    byte__11 = self.round_keys[i__9 - 4][j__11
                        ] ^ self.round_keys[i__9 - 1][j__11]
                    self.round_keys[i__9].append(byte__11)
                    print('exit scope 13')
            print('exit scope 11')
            print('exit scope 10')
        print('exit scope 9')

    def encrypt(self, plaintext):
        print('enter scope 14')
        print(1, 611)
        self = self
        plaintext__14 = plaintext
        print(43, 612)
        self.plain_state = text2matrix(plaintext__14)
        self.__add_round_key(self.plain_state, self.round_keys[:4])
        for i__14 in range(1, 10):
            print('enter scope 15')
            self.__round_encrypt(self.plain_state, self.round_keys[4 *
                i__14:4 * (i__14 + 1)])
            print('exit scope 15')
        self.__sub_bytes(self.plain_state)
        self.__shift_rows(self.plain_state)
        self.__add_round_key(self.plain_state, self.round_keys[40:])
        print('exit scope 14')
        return matrix2text(self.plain_state)
        print('exit scope 14')

    def decrypt(self, ciphertext):
        print('enter scope 16')
        print(1, 625)
        self = self
        ciphertext__16 = ciphertext
        print(50, 626)
        self.cipher_state = text2matrix(ciphertext__16)
        self.__add_round_key(self.cipher_state, self.round_keys[40:])
        self.__inv_shift_rows(self.cipher_state)
        self.__inv_sub_bytes(self.cipher_state)
        for i__16 in range(9, 0, -1):
            print('enter scope 17')
            self.__round_decrypt(self.cipher_state, self.round_keys[4 *
                i__16:4 * (i__16 + 1)])
            print('exit scope 17')
        self.__add_round_key(self.cipher_state, self.round_keys[:4])
        print('exit scope 16')
        return matrix2text(self.cipher_state)
        print('exit scope 16')

    def __add_round_key(self, s, k):
        print('enter scope 18')
        print(1, 641)
        self = self
        s__18 = s
        k__18 = k
        for i__18 in range(4):
            print('enter scope 19')
            for j__19 in range(4):
                print('enter scope 20')
                print(60, 644)
                s__18[i__18][j__19] ^= k__18[i__18][j__19]
                print('exit scope 20')
            print('exit scope 19')
        print('exit scope 18')

    def __round_encrypt(self, state_matrix, key_matrix):
        print('enter scope 21')
        print(1, 646)
        self = self
        state_matrix__21 = state_matrix
        key_matrix__21 = key_matrix
        self.__sub_bytes(state_matrix__21)
        self.__shift_rows(state_matrix__21)
        self.__mix_columns(state_matrix__21)
        self.__add_round_key(state_matrix__21, key_matrix__21)
        print('exit scope 21')

    def __round_decrypt(self, state_matrix, key_matrix):
        print('enter scope 22')
        print(1, 652)
        self = self
        state_matrix__22 = state_matrix
        key_matrix__22 = key_matrix
        self.__add_round_key(state_matrix__22, key_matrix__22)
        self.__inv_mix_columns(state_matrix__22)
        self.__inv_shift_rows(state_matrix__22)
        self.__inv_sub_bytes(state_matrix__22)
        print('exit scope 22')

    def __sub_bytes(self, s):
        print('enter scope 23')
        print(1, 658)
        self = self
        s__23 = s
        for i__23 in range(4):
            print('enter scope 24')
            for j__24 in range(4):
                print('enter scope 25')
                print(73, 661)
                s__23[i__23][j__24] = Sbox__0[s__23[i__23][j__24]]
                print('exit scope 25')
            print('exit scope 24')
        print('exit scope 23')

    def __inv_sub_bytes(self, s):
        print('enter scope 26')
        print(1, 663)
        self = self
        s__26 = s
        for i__26 in range(4):
            print('enter scope 27')
            for j__27 in range(4):
                print('enter scope 28')
                print(80, 666)
                s__26[i__26][j__27] = InvSbox__0[s__26[i__26][j__27]]
                print('exit scope 28')
            print('exit scope 27')
        print('exit scope 26')

    def __shift_rows(self, s):
        print('enter scope 29')
        print(1, 668)
        self = self
        s__29 = s
        print(84, 669)
        s__29[0][1], s__29[1][1], s__29[2][1], s__29[3][1] = s__29[1][1
            ], s__29[2][1], s__29[3][1], s__29[0][1]
        print(84, 670)
        s__29[0][2], s__29[1][2], s__29[2][2], s__29[3][2] = s__29[2][2
            ], s__29[3][2], s__29[0][2], s__29[1][2]
        print(84, 671)
        s__29[0][3], s__29[1][3], s__29[2][3], s__29[3][3] = s__29[3][3
            ], s__29[0][3], s__29[1][3], s__29[2][3]
        print('exit scope 29')

    def __inv_shift_rows(self, s):
        print('enter scope 30')
        print(1, 673)
        self = self
        s__30 = s
        print(87, 674)
        s__30[0][1], s__30[1][1], s__30[2][1], s__30[3][1] = s__30[3][1
            ], s__30[0][1], s__30[1][1], s__30[2][1]
        print(87, 675)
        s__30[0][2], s__30[1][2], s__30[2][2], s__30[3][2] = s__30[2][2
            ], s__30[3][2], s__30[0][2], s__30[1][2]
        print(87, 676)
        s__30[0][3], s__30[1][3], s__30[2][3], s__30[3][3] = s__30[1][3
            ], s__30[2][3], s__30[3][3], s__30[0][3]
        print('exit scope 30')

    def __mix_single_column(self, a):
        print('enter scope 31')
        print(1, 678)
        self = self
        a__31 = a
        print(90, 680)
        t__31 = a__31[0] ^ a__31[1] ^ a__31[2] ^ a__31[3]
        print(90, 681)
        u__31 = a__31[0]
        print(90, 682)
        a__31[0] ^= t__31 ^ xtime(a__31[0] ^ a__31[1])
        print(90, 683)
        a__31[1] ^= t__31 ^ xtime(a__31[1] ^ a__31[2])
        print(90, 684)
        a__31[2] ^= t__31 ^ xtime(a__31[2] ^ a__31[3])
        print(90, 685)
        a__31[3] ^= t__31 ^ xtime(a__31[3] ^ u__31)
        print('exit scope 31')

    def __mix_columns(self, s):
        print('enter scope 32')
        print(1, 687)
        self = self
        s__32 = s
        for i__32 in range(4):
            print('enter scope 33')
            self.__mix_single_column(s__32[i__32])
            print('exit scope 33')
        print('exit scope 32')

    def __inv_mix_columns(self, s):
        print('enter scope 34')
        print(1, 691)
        self = self
        s__34 = s
        for i__34 in range(4):
            print('enter scope 35')
            print(99, 694)
            u__35 = xtime(xtime(s__34[i__34][0] ^ s__34[i__34][2]))
            print(99, 695)
            v__35 = xtime(xtime(s__34[i__34][1] ^ s__34[i__34][3]))
            print(99, 696)
            s__34[i__34][0] ^= u__35
            print(99, 697)
            s__34[i__34][1] ^= v__35
            print(99, 698)
            s__34[i__34][2] ^= u__35
            print(99, 699)
            s__34[i__34][3] ^= v__35
            print('exit scope 35')
        self.__mix_columns(s__34)
        print('exit scope 34')


print('enter scope 36')
if __name__ == '__main__':
    import time
    print(102, 705)
    start__36 = time.time()
    print(102, 706)
    aes__36 = AES(1212304810341341)
    aes__36.encrypt(1212304810341341)
    print(102, 708)
    end__36 = time.time()
print('exit scope 36')
